"""Bounded GT-local recovery with exact labels, prior-grid reuse and restart support."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

from .gt_neighborhood import local_keys, relative_heading, experiment_protocol, run_definition
from .high_score_bank import COUNT, THRESHOLD, select_indices, save_bank, read_scene
from .retained_bank import publish_full_scene, read_full_scene
from .scene_store import SceneTable, row_digest, validate_rows
from .store import atomic, checksum, digest, lock, read_json, write_json


def refinement_keys(rows, gt_score, config, unit):
    """Refine near GT and diverse promising measured endpoints, within the GT box."""
    score = rows["labels"][:, -1]
    if max(float(score.max()), float(gt_score)) < config["refine_min_raw"]:
        return np.empty((0, 2), dtype=np.int32)
    seen = {tuple(k) for k in rows["keys"]}
    result = []
    bound = round(config["max_offset_m"] / unit)
    remaining = config["max_queries"] - len(rows)

    def add(points):
        for x, y in points:
            k = (int(x), int(y))
            if max(abs(k[0]), abs(k[1])) <= bound and k not in seen:
                seen.add(k)
                result.append(k)
                if len(result) >= remaining:
                    return True
        return False

    # A centered micro-grid preserves access to the exact GT neighborhood.
    if add(local_keys(config["micro_rings_m"], unit)):
        return np.asarray(result, dtype=np.int32)
    ranked = np.flatnonzero(score >= config["refine_min_raw"])
    ranked = ranked[np.argsort(-score[ranked], kind="stable")]
    seeds = []
    for i in ranked:
        k = rows["keys"][i]
        if (
            not seeds
            or min(np.linalg.norm((k - s) * unit) for s in seeds) >= config["seed_separation_m"]
        ):
            seeds.append(k.copy())
            if len(seeds) == config["max_seeds"]:
                break
    offsets = local_keys(config["seed_rings_m"], unit)
    # Round-robin offsets give every selected region some queries before the cap.
    for offset in offsets:
        if add([seed + offset for seed in seeds]):
            break
    return np.asarray(result, dtype=np.int32).reshape(-1, 2)


def score_request(table, keys, gt, zero, protocol, evaluator):
    from .source import construct

    if table.state["pending"] is None:
        table.begin(dict(name="gt_recovery_%06d" % len(table.rows), keys=keys.tolist()))
    pending = table.state["pending"]
    batch = np.asarray(pending["keys"][pending["offset"] :], dtype=np.int32).reshape(-1, 2)
    if len(batch):
        paths = relative_heading(construct(gt, batch, protocol)[0], zero, gt, batch)
        labels = np.stack([evaluator.score(t) for t in paths])
        table.append(batch, paths, labels)
    table.end()


def reuse_pilot(table, seed, keys, gt, zero, protocol, metadata, evaluator):
    """Reuse matching base-grid labels only after identity and construction checks."""
    from .source import construct

    if seed is None or not (seed / "ALL_SCORES.json").exists():
        return 0
    identity, old = read_full_scene(seed)
    source_protocol = read_json(seed / "contract.json")["protocol"]
    if identity != metadata or source_protocol["constructor"] != protocol["constructor"]:
        raise ValueError("Prior pilot identity or constructor differs")
    if source_protocol["labeling"] != protocol["labeling"] or len(old["keys"]) != len(keys):
        raise ValueError("Prior pilot labeling or grid length differs")
    np.testing.assert_array_equal(old["keys"], keys)
    for offset in range(0, len(keys), 64):
        b = keys[offset : offset + 64]
        paths = relative_heading(construct(gt, b, protocol)[0], zero, gt, b)
        np.testing.assert_array_equal(paths, old["trajectories"][offset : offset + len(b)])
    for i in [0, len(keys) // 2, len(keys) - 1]:
        checked = evaluator.score(old["trajectories"][i])
        if (
            np.max(abs(checked - old["labels"][i]))
            > protocol["labeling"]["float64_parity_tolerance"]
        ):
            raise ValueError("Prior pilot official replay mismatch")
    offset = len(table.rows)
    for k in ("keys", "trajectories", "labels"):
        np.testing.assert_array_equal(table.rows[k], old[k][:offset])
    while len(table.rows) < len(keys) or table.state["pending"] is not None:
        if table.state["pending"] is None:
            start = len(table.rows)
            table.begin(dict(name="reuse_%06d" % start, keys=keys[start : start + 64].tolist()))
        pending = table.state["pending"]
        n = len(pending["keys"]) - pending["offset"]
        start = len(table.rows)
        if n:
            block = slice(start, start + n)
            np.testing.assert_array_equal(old["keys"][block], pending["keys"][pending["offset"] :])
            table.append(old["keys"][block], old["trajectories"][block], old["labels"][block])
        table.end()
    write_json(
        table.directory / "REUSED.json",
        dict(
            source=str(seed),
            rows=len(keys),
            source_marker=checksum(seed / "ALL_SCORES.json"),
            replayed_labels=3,
        ),
    )
    return len(keys)


def run_scene(item, config, output, run_id):
    import torch
    from .source import load_scene, construct, Evaluator

    torch.set_num_threads(1)
    record = item["record"]
    source = Path(config["source"]) / "scenes" / record["split"] / record["token"]
    metadata = read_json(source / "contract.json")["identity"]
    original = read_json(source / "contract.json")["protocol"]
    _, gt, cache, current = load_scene(record, [record["token"]], original)
    if current != metadata:
        raise ValueError("Source/cache identity mismatch")
    unit = original["sampling"]["integer_lattice_m"]
    base = local_keys(config["base_rings_m"], unit)
    protocol = experiment_protocol(original, "gt_relative_heading", base)
    protocol["dataset_version"] = "gt_neighborhood_recovery_v1"
    protocol["sampling"].update(
        method="gt_local_base_then_threshold_refinement",
        max_unique_goals=config["max_queries"],
        query_budgets=[len(base), config["max_queries"]],
        recovery_settings={
            k: config[k]
            for k in (
                "base_rings_m",
                "micro_rings_m",
                "seed_rings_m",
                "refine_min_raw",
                "max_seeds",
                "seed_separation_m",
                "max_offset_m",
            )
        },
        stop_rule="finish_base_then_stop_when_256_distinct_raw_pdm_ge_0.95",
    )
    directory = Path(output) / "scenes" / record["split"] / record["token"]
    contract = dict(
        identity=metadata, protocol=protocol, source_protocol=original, run_contract=run_id
    )
    with lock(directory):
        if (directory / "RESULT.json").exists():
            if read_json(directory / "contract.json") != contract:
                raise ValueError("Completed scene contract changed")
            read_full_scene(directory)
            result = read_json(directory / "RESULT.json")
            if result["passed"]:
                read_scene(directory, verify=True)
            return result
        table = SceneTable(directory, contract, config["max_queries"])
        try:
            zero = construct(gt, np.zeros((64, 2), dtype=np.int32), protocol)[0][0]
            evaluator = Evaluator(cache)
            gt_label = evaluator.score(gt.astype(np.float32))
            write_json(directory / "expert.json", dict(labels=gt_label.tolist()))
            np.testing.assert_array_equal(
                table.rows["keys"][: len(base)], base[: min(len(table.rows), len(base))]
            )
            seed = Path(config["reuse_root"]) / "scenes" / record["split"] / record["token"]
            if len(table.rows) < len(base):
                reuse_pilot(table, seed, base, gt, zero, protocol, metadata, evaluator)
            while len(table.rows) < len(base):
                score_request(
                    table,
                    base[len(table.rows) : len(table.rows) + 64],
                    gt,
                    zero,
                    protocol,
                    evaluator,
                )
            # Resolve a pending block after a crash, even if its scores already satisfy K.
            if table.state["pending"] is not None:
                score_request(table, np.empty((0, 2), np.int32), gt, zero, protocol, evaluator)
            plan_path = directory / "refinement_plan.json"
            base_digest = row_digest(table.rows[: len(base)])
            if plan_path.exists():
                plan = read_json(plan_path)
                if plan["base_digest"] != base_digest:
                    raise ValueError("Refinement base changed")
            else:
                extra = refinement_keys(table.rows[: len(base)], gt_label[-1], config, unit)
                plan = dict(base_digest=base_digest, keys=extra.tolist())
                write_json(plan_path, plan)
            all_keys = np.concatenate((base, np.asarray(plan["keys"], np.int32).reshape(-1, 2)))
            np.testing.assert_array_equal(table.rows["keys"], all_keys[: len(table.rows)])
            while len(table.rows) < len(all_keys):
                if (
                    len(select_indices(table.rows["trajectories"], table.rows["labels"], gt))
                    == COUNT
                ):
                    break
                score_request(
                    table,
                    all_keys[len(table.rows) : len(table.rows) + 64],
                    gt,
                    zero,
                    protocol,
                    evaluator,
                )
            validate_rows(table.rows, gt, protocol)
            ids = select_indices(table.rows["trajectories"], table.rows["labels"], gt)
            qualified = select_indices(
                table.rows["trajectories"], table.rows["labels"], gt, count=len(table.rows)
            )
            result = dict(
                token=record["token"],
                split=record["split"],
                format="high_score_256",
                passed=len(ids) == COUNT,
                candidates=len(ids),
                queries=len(table.rows),
                base_queries=len(base),
                planned_queries=len(all_keys),
                distinct_qualified=len(qualified),
                best_pdms=float(table.rows["labels"][:, -1].max() * 100),
                gt_pdms=float(gt_label[-1] * 100),
                completed_unix=time.time(),
            )
            if len(ids) == COUNT:
                audit = Evaluator(cache)
                checked = np.stack(
                    [audit.score(table.rows["trajectories"][i], details=True)[0] for i in ids]
                )
                error = float(abs(checked - table.rows["labels"][ids]).max())
                if (
                    error > protocol["labeling"]["float64_parity_tolerance"]
                    or (checked[:, -1] < THRESHOLD).any()
                ):
                    raise ValueError("Official bank verification failed")
                _, bank = save_bank(directory, table.rows, gt, protocol)
                xy = table.rows["trajectories"][ids, -1, :2]
                result.update(
                    bank,
                    audited_candidates=COUNT,
                    max_official_error=error,
                    min_selected_pdms=float(checked[:, -1].min() * 100),
                    selected_endpoint_span_m=np.ptp(xy, axis=0).tolist(),
                )
            publish_full_scene(directory, len(table.rows))
            write_json(
                directory / ("COMPLETE.json" if result["passed"] else "SHORTFALL.json"), result
            )
            if result["passed"]:
                read_scene(directory, verify=True)
            write_json(directory / "RESULT.json", result)
            return result
        finally:
            table.close()


def publish_results(config, output, summary):
    """Use the verified fixed-size replacement transaction after all scenes finish."""
    from .merge_recovered import prepare_plan, prepare_scene, install_scene, finalize

    if summary["errors"] or summary["processed"] != summary["total"]:
        raise ValueError("Finish error-free recovery before publication")
    if not summary["completed"]:
        write_json(
            Path(output) / "publication.json", dict(replaced_scenes=0, source_unchanged=True)
        )
        return
    merge = dict(
        target=config["source"],
        sources=[str(output)],
        revision=config["publication_revision"],
        expected_scenes=summary["completed"],
        full_count=3072,
    )
    write_json(Path(output) / "publication_config.json", merge)
    with lock(config["source"]):
        plan = prepare_plan(merge)
        for item in plan["scenes"]:
            prepare_scene(merge, item)
        for item in plan["scenes"]:
            install_scene(merge, item)
        result = finalize(merge, plan)
    write_json(Path(output) / "publication.json", result)
    print(
        json.dumps(
            dict(
                publication_complete=True,
                replaced_scenes=result["replaced_scenes"],
                high_scenes=result["completed_scenes"],
                remaining=result["shortfall_scenes"],
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "run"))
    parser.add_argument("--config", type=Path, default=Path("feedback/configs/gt_recovery551.json"))
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Merge audited successes into the original fixed-size bank",
    )
    args = parser.parse_args()
    config = read_json(args.config)
    items = read_json(Path(config["cohort"]))
    if args.pilot:
        items = sorted(items, key=lambda x: (-x["gt_pdms"], -x["eligible"], x["token"]))[
            : args.pilot
        ]
    output = (args.output or Path(config["output"])).resolve()
    source = Path(config["source"]).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Recovery requires a separate output directory")
    if (
        args.workers < 1
        or args.pilot < 0
        or not items
        or len({x["token"] for x in items}) != len(items)
    ):
        raise ValueError("Invalid cohort/workers")
    if args.publish and (args.pilot or output != Path(config["output"]).resolve()):
        raise ValueError("Publication is restricted to the complete configured cohort/output")
    run = run_definition(config, items)
    print(
        json.dumps(
            dict(
                scenes=len(items),
                workers=args.workers,
                output=str(output),
                base_queries=len(local_keys(config["base_rings_m"], 0.001)),
                max_queries=config["max_queries"],
            )
        ),
        flush=True,
    )
    if args.command == "inspect":
        return
    with lock(output):
        if (output / "run.json").exists() and read_json(output / "run.json") != run:
            raise ValueError("Run inputs or code changed")
        write_json(output / "run.json", run)
        if (output / "summary.json").exists():
            finished = read_json(output / "summary.json")
            if (
                finished["state"] == "finished"
                and not finished["errors"]
                and finished["processed"] == len(items)
            ):
                if args.publish:
                    publish_results(config, output, finished)
                return
        atomic(
            output / "scenes.jsonl",
            lambda f: f.write("".join(json.dumps(x["record"]) + "\n" for x in items).encode()),
        )
        results, errors = [], []
        start = time.time()
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(run_scene, x, config, str(output), digest(run)): x for x in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    write_json(output / "results" / (item["token"] + ".json"), result)
                    print(json.dumps(result), flush=True)
                except Exception:
                    error = dict(token=item["token"], traceback=traceback.format_exc())
                    errors.append(error)
                    write_json(output / "errors" / (item["token"] + ".json"), error)
                    print(json.dumps(error), flush=True)
                write_json(
                    output / "status.json",
                    dict(
                        state="running",
                        total=len(items),
                        processed=len(results),
                        completed=sum(r["passed"] for r in results),
                        scored=sum(r["queries"] for r in results),
                        errors=errors,
                        elapsed_s=time.time() - start,
                        updated_unix=time.time(),
                    ),
                )
        complete = sorted(r["token"] for r in results if r["passed"])
        remaining = sorted(x["token"] for x in items if x["token"] not in set(complete))
        summary = dict(
            state="finished",
            total=len(items),
            processed=len(results),
            completed=len(complete),
            recovered_tokens=complete,
            remaining_tokens=remaining,
            errors=errors,
            scored=sum(r["queries"] for r in results),
            elapsed_s=time.time() - start,
            updated_unix=time.time(),
            passed=len(complete) == len(items),
        )
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", summary)
        write_json(output / ("COMPLETE.json" if summary["passed"] else "PARTIAL.json"), summary)
        atomic(
            output / "remaining_shortfall_tokens.txt",
            lambda f: f.write(("\n".join(remaining) + "\n").encode()),
        )
        print(
            json.dumps({k: v for k, v in summary.items() if not k.endswith("tokens")}), flush=True
        )
        if errors:
            raise RuntimeError("Recovery has scene errors")
        if args.publish:
            publish_results(config, output, summary)


if __name__ == "__main__":
    main()
