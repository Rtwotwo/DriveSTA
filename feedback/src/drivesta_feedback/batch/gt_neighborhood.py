"""Paired GT-neighborhood experiment with independently readable scored banks."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

from .high_score_bank import COUNT, THRESHOLD, select_indices, save_bank, read_scene
from .retained_bank import publish_full_scene, read_full_scene
from .scene_store import SceneTable, validate_rows
from .store import atomic, checksum, digest, lock, read_json, write_json

ARMS = ("tto_heading", "gt_relative_heading")


def experiment_protocol(source_protocol, arm, keys):
    protocol = copy.deepcopy(source_protocol)
    protocol["dataset_version"] = "gt_neighborhood_paired_v1"
    protocol["constructor"]["gt_neighborhood_heading"] = arm
    protocol["sampling"] = dict(
        method="nested_gt_endpoint_grid",
        integer_lattice_m=source_protocol["sampling"]["integer_lattice_m"],
        max_unique_goals=len(keys),
        query_budgets=[len(keys)],
        high_pdm_raw=THRESHOLD,
        keys_digest=digest(keys.tolist()),
    )
    protocol["storage"] = dict(
        chunk_candidates=64,
        retain_all_scores=True,
        format="all_scored_candidates_v1_plus_high_score_256",
        candidates=COUNT,
    )
    protocol["execution"] = dict(
        parallel_unit="scene", threads_per_worker=1, scene_workers="recorded_in_launch_receipt"
    )
    protocol["training"] = dict(launch_training=False, online_pdm_calls=False)
    return protocol


def run_definition(config, items):
    paths = set(Path(__file__).parent.glob("*.py"))
    paths.update(Path("navsim/planning/simulation/planner/pdm_planner").rglob("*.py"))
    paths.update(Path("navsim/planning/metric_caching").glob("*.py"))
    paths.update(
        [
            Path("navsim/evaluate/pdm_score.py"),
            Path("navsim/tools/trajectory_mining/path_endpoint_surface.py"),
            Path("navsim/common/dataclasses.py"),
            Path("navsim/common/dataloader.py"),
            Path("feedback/configs/shortfall32348_env.json"),
        ]
    )
    return dict(
        metadata_version=2,
        config=config,
        cohort=items,
        implementations={str(p): checksum(p) for p in sorted(paths)},
    )


def local_keys(rings, unit):
    """Nested square grids around the expert endpoint, measured in ego XY."""
    seen, result = set(), []
    for radius, spacing in rings:
        n = round(radius / spacing)
        if n < 1 or abs(n * spacing - radius) > 1e-9:
            raise ValueError("Radius must be a positive integer multiple of spacing")
        values = np.rint(np.arange(-n, n + 1) * spacing / unit).astype(np.int32)
        ring = [(int(x), int(y)) for x in values for y in values]
        for key in sorted(ring, key=lambda p: (p[0] ** 2 + p[1] ** 2, p)):
            if key not in seen:
                seen.add(key)
                result.append(key)
    return np.asarray(result, dtype=np.int32)


def relative_heading(paths, zero_path, gt, keys):
    """Keep GT heading at zero and transfer only the warp's angular change."""
    result = paths.copy()
    change = paths[..., 2].astype(np.float64) - zero_path[None, :, 2]
    change = np.arctan2(np.sin(change), np.cos(change))
    angle = np.asarray(gt)[None, :, 2] + change
    result[..., 2] = np.arctan2(np.sin(angle), np.cos(angle))
    result[np.all(np.asarray(keys) == 0, axis=1)] = np.asarray(gt, dtype=np.float32)
    return result


def choose_cohort(items, pilot):
    if not pilot:
        return items
    priority = [x for x in items if x["gt_pdms"] >= 95]
    priority += [x for x in items if x["eligible"] > 0 and x not in priority]
    rest = sorted([x for x in items if x not in priority], key=lambda x: (x["gt_pdms"], x["token"]))
    left = max(0, pilot - len(priority))
    if left:
        priority += [rest[i] for i in np.linspace(0, len(rest) - 1, left).round().astype(int)]
    return priority[:pilot]


def run_arm(
    record, source, output, protocol, metadata, gt, evaluator, keys, zero_path, arm, run_id
):
    from .source import construct, Evaluator

    directory = Path(output) / arm / "scenes" / record["split"] / record["token"]
    source_protocol = copy.deepcopy(protocol)
    protocol = experiment_protocol(protocol, arm, keys)
    contract = dict(
        identity=metadata, protocol=protocol, source_protocol=source_protocol, run_contract=run_id
    )
    with lock(directory):
        result_file = directory / "RESULT.json"
        if result_file.exists():
            if read_json(directory / "contract.json") != contract:
                raise ValueError("Completed GT-neighborhood contract differs")
            read_full_scene(directory)
            result = read_json(result_file)
            if result["passed"]:
                read_scene(directory, verify=True)
            return result
        table = SceneTable(directory, contract, len(keys))
        try:
            if not np.array_equal(table.rows["keys"], keys[: len(table.rows)]):
                raise ValueError("Resumed candidate prefix differs from deterministic grid")
            # Full fixed-grid comparison: neither arm stops early on finding 256.
            while len(table.rows) < len(keys) or table.state["pending"] is not None:
                if table.state["pending"] is None:
                    offset = len(table.rows)
                    table.begin(
                        dict(
                            name="gt_local_%06d" % offset, keys=keys[offset : offset + 64].tolist()
                        )
                    )
                pending = table.state["pending"]
                batch = np.asarray(pending["keys"][pending["offset"] :], dtype=np.int32)
                if len(batch):
                    paths, _ = construct(gt, batch, protocol)
                    if arm == "gt_relative_heading":
                        paths = relative_heading(paths, zero_path, gt, batch)
                    labels = np.stack([evaluator.score(t) for t in paths])
                    table.append(batch, paths, labels)
                table.end()
            validate_rows(table.rows, gt, protocol)
            ids = select_indices(table.rows["trajectories"], table.rows["labels"], gt)
            all_ids = select_indices(
                table.rows["trajectories"], table.rows["labels"], gt, count=len(keys)
            )
            union_ids = select_indices(
                np.concatenate((source["trajectories"], table.rows["trajectories"])),
                np.concatenate((source["labels"], table.rows["labels"])),
                gt,
            )
            score = table.rows["labels"][:, -1]
            result = dict(
                token=record["token"],
                split=record["split"],
                arm=arm,
                format="high_score_256",
                passed=len(ids) == COUNT,
                candidates=len(ids),
                queries=len(keys),
                qualified_queries=int((score >= THRESHOLD).sum()),
                distinct_qualified=len(all_ids),
                source_plus_new_qualified_up_to_256=len(union_ids),
                best_pdms=float(score.max() * 100),
                min_selected_pdms=float(score[ids].min() * 100) if len(ids) else None,
                source_rows=len(source["labels"]),
                completed_unix=time.time(),
            )
            if len(ids) == COUNT:
                audit = Evaluator(evaluator.cache)
                checked = np.stack(
                    [audit.score(table.rows["trajectories"][i], details=True)[0] for i in ids]
                )
                error = float(abs(checked - table.rows["labels"][ids]).max())
                if (
                    error > protocol["labeling"]["float64_parity_tolerance"]
                    or (checked[:, -1] < THRESHOLD).any()
                ):
                    raise ValueError("Selected trajectory failed official verification")
                _, bank = save_bank(directory, table.rows, gt, protocol)
                selected_xy = table.rows["trajectories"][ids, -1, :2]
                result.update(
                    bank,
                    audited_candidates=len(ids),
                    max_official_error=error,
                    selected_endpoint_span_m=np.ptp(selected_xy, axis=0).tolist(),
                )
            publish_full_scene(directory, len(table.rows))
            write_json(
                directory / ("COMPLETE.json" if result["passed"] else "SHORTFALL.json"), result
            )
            if result["passed"]:
                read_scene(directory, verify=True)
            write_json(result_file, result)
            return result
        finally:
            table.close()


def run_scene(item, config, output, run_id):
    import torch
    from .source import load_scene, construct, Evaluator

    torch.set_num_threads(1)
    record = item["record"]
    src = Path(config["source"]) / "scenes" / record["split"] / record["token"]
    source_marker = checksum(src / "ALL_SCORES.json")
    metadata, source = read_full_scene(src)
    protocol = read_json(src / "contract.json")["protocol"]
    _, gt, cache, current = load_scene(record, [record["token"]], protocol)
    if current != metadata:
        raise ValueError("Source and newly loaded scene/cache identities differ")
    keys = local_keys(config["rings_m"], protocol["sampling"]["integer_lattice_m"])
    zero_path = construct(gt, np.zeros((64, 2), dtype=np.int32), protocol)[0][0]
    evaluator = Evaluator(cache)
    result = dict(
        token=record["token"],
        split=record["split"],
        original_gt_pdms=item["gt_pdms"],
        original_eligible=item["eligible"],
        original_best_pdms=item["best_pdms"],
        gt_official=evaluator.score(gt).tolist(),
        gt_float32_official=evaluator.score(gt.astype(np.float32)).tolist(),
        zero_warp_official=evaluator.score(zero_path).tolist(),
        max_zero_heading_error_rad=float(
            abs(
                np.arctan2(np.sin(zero_path[:, 2] - gt[:, 2]), np.cos(zero_path[:, 2] - gt[:, 2]))
            ).max()
        ),
        arms={},
    )
    for arm in ARMS:
        result["arms"][arm] = run_arm(
            record, source, output, protocol, metadata, gt, evaluator, keys, zero_path, arm, run_id
        )
    if checksum(src / "ALL_SCORES.json") != source_marker:
        raise ValueError("Source full-score publication changed during experiment")
    write_json(Path(output) / "results" / (record["token"] + ".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "run"))
    parser.add_argument(
        "--config", type=Path, default=Path("feedback/configs/gt_neighborhood.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot", type=int, default=16, help="0 selects all remaining scenes")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    config = read_json(args.config)
    items = choose_cohort(read_json(Path(config["cohort"])), args.pilot)
    if args.workers < 1 or args.pilot < 0 or not items:
        raise ValueError("Invalid workers/cohort size")
    source = Path(config["source"]).resolve()
    output = (args.output or Path(config["output"])).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Experiment output must be separate from the source")
    run = run_definition(config, items)
    run_id = digest(run)
    if (output / "run.json").exists() and read_json(output / "run.json") != run:
        raise ValueError("Run inputs or implementation changed; refusing to mix results")
    info = dict(
        scenes=len(items),
        workers=args.workers,
        arms=ARMS,
        candidates_per_scene_per_arm=len(local_keys(config["rings_m"], 0.001)),
        output=str(output),
    )
    print(json.dumps(info), flush=True)
    if args.command == "inspect":
        return
    with lock(output):
        write_json(output / "run.json", run)
        manifest = "".join(json.dumps(x["record"]) + "\n" for x in items).encode()
        for arm in ARMS:
            atomic(output / arm / "scenes.jsonl", lambda f: f.write(manifest))
        results, errors = [], []
        start = time.time()
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(run_scene, x, config, str(output), run_id): x["token"] for x in items
            }
            for future in as_completed(futures):
                token = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(json.dumps(result), flush=True)
                except Exception:
                    error = dict(token=token, traceback=traceback.format_exc())
                    errors.append(error)
                    write_json(output / "errors" / (token + ".json"), error)
                    print(json.dumps(error), flush=True)
                write_json(
                    output / "status.json",
                    dict(
                        total=len(items),
                        processed=len(results),
                        errors=errors,
                        updated_unix=time.time(),
                        elapsed_s=time.time() - start,
                    ),
                )
        summary = dict(
            total=len(items),
            processed=len(results),
            errors=errors,
            completed_unix=time.time(),
            elapsed_s=time.time() - start,
            arms={},
        )
        recovered = set()
        for arm in ARMS:
            complete = [r["token"] for r in results if r["arms"][arm]["passed"]]
            recovered.update(complete)
            summary["arms"][arm] = dict(
                complete=len(complete),
                complete_tokens=complete,
                qualified_scenes=sum(r["arms"][arm]["distinct_qualified"] > 0 for r in results),
                qualified_trajectories=sum(r["arms"][arm]["distinct_qualified"] for r in results),
            )
            write_json(
                output / arm / ("COMPLETE.json" if len(complete) == len(items) else "PARTIAL.json"),
                dict(
                    passed=len(complete) == len(items),
                    completed=len(complete),
                    requested=len(items),
                ),
            )
        summary["recovered_tokens"] = sorted(recovered)
        summary["remaining_tokens"] = [x["token"] for x in items if x["token"] not in recovered]
        write_json(output / "summary.json", summary)
        atomic(
            output / "remaining_tokens.txt",
            lambda f: f.write(("\n".join(summary["remaining_tokens"]) + "\n").encode()),
        )
        print(json.dumps(summary), flush=True)
        if errors:
            raise RuntimeError("GT-neighborhood experiment contains errors")


from scipy.interpolate import CubicSpline
from .filter_targets import wrap


def neighborhood_parameters(count, seed):
    """One-axis lines first, followed by deterministic multiscale combinations."""
    from .adaptive_sampling import shuffled_parameters

    rows = []
    for magnitude in np.linspace(1 / 32, 1.0, 32):
        for axis in range(6):
            for sign in (-1.0, 1.0):
                row = np.zeros(6)
                row[axis] = sign * magnitude
                rows.append(row)
    lines = np.asarray(rows)
    if count <= len(lines):
        return lines[:count]
    combinations = shuffled_parameters(count - len(lines), seed)
    scales = np.asarray([0.025, 0.075, 0.2, 0.5, 1.0])
    combinations *= scales[np.arange(len(combinations)) % len(scales), None]
    return np.concatenate([lines, combinations])


def gt_local_proposals(reference, coefficients):
    """C2 offsets and monotone timing within the recorded four-second window.

    Six coefficients are: timing, terminal along/across, middle along/across,
    and early/late lateral offset. Heading follows the change in curve tangent;
    it is not independently perturbed. Zero coefficients reproduce the reference.
    """
    reference = np.asarray(reference, np.float32)
    c = np.asarray(coefficients, float)
    if reference.shape != (8, 3) or c.ndim != 2 or c.shape[1] != 6:
        raise ValueError("Expected reference[8,3] and coefficients[N,6]")
    if not np.isfinite(reference).all() or not np.isfinite(c).all():
        raise ValueError("Non-finite GT proposal input")
    if not len(c):
        return np.empty((0, 8, 3), np.float32)
    knots = np.arange(9) * 0.5
    path = np.vstack([np.zeros(3), reference])
    xy = CubicSpline(knots, path[:, :2], axis=0)
    heading = CubicSpline(knots, np.unwrap(path[:, 2]))

    def basis(t):
        u = t / 4
        terminal = u**3 * (10 - 15 * u + 6 * u * u)
        middle = 64 * u**3 * (1 - u) ** 3
        early_late = middle * (2 * u - 1)
        dt = (30 * u * u - 60 * u**3 + 30 * u**4) / 4
        dm = 48 * u * u * (1 - u) ** 2 * (1 - 2 * u)
        de = dm * (2 * u - 1) + middle / 2
        return terminal, middle, early_late, dt, dm, de

    dense = np.linspace(0, 4, 161)
    _, mid, _, _, dmid, _ = basis(dense)
    times = dense[None] + c[:, :1] * mid
    if np.any(1 + c[:, :1] * dmid <= 0) or times.min() < 0 or times.max() > 4:
        raise ValueError("Timing must be monotone and remain inside observed GT")
    t = knots[1:]
    end, mid, early, dend, dmid, dearly = basis(t)
    desired = t[None] + c[:, :1] * mid
    rate = 1 + c[:, :1] * dmid
    base_xy = xy(desired)
    base_h = heading(desired)
    base_v = xy(desired, 1) * rate[..., None]
    h_rate = heading(desired, 1) * rate
    tangent = np.stack([np.cos(base_h), np.sin(base_h)], -1)
    normal = np.stack([-tangent[..., 1], tangent[..., 0]], -1)
    along = c[:, 1:2] * end + c[:, 3:4] * mid
    across = c[:, 2:3] * end + c[:, 4:5] * mid + c[:, 5:6] * early
    dalong = c[:, 1:2] * dend + c[:, 3:4] * dmid
    dacross = c[:, 2:3] * dend + c[:, 4:5] * dmid + c[:, 5:6] * dearly
    offset = along[..., None] * tangent + across[..., None] * normal
    delta_v = (dalong - across * h_rate)[..., None] * tangent + (dacross + along * h_rate)[
        ..., None
    ] * normal
    changed_v = base_v + delta_v
    base_angle = np.where(
        np.linalg.norm(base_v, axis=-1) >= 0.05, np.arctan2(base_v[..., 1], base_v[..., 0]), base_h
    )
    correction = wrap(np.arctan2(changed_v[..., 1], changed_v[..., 0]) - base_angle)
    correction = np.where(np.linalg.norm(changed_v, axis=-1) >= 0.05, correction, 0.0)
    result = np.concatenate([base_xy + offset, wrap(base_h + correction)[..., None]], -1).astype(
        np.float32
    )
    result[(c == 0).all(1)] = reference
    return result


if __name__ == "__main__":
    main()
