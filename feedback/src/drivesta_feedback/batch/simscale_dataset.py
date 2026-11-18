"""Resumable SimScale full3072 / physically screened high256 production."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np

from .store import atomic, write_json, read_json, write_npz, checksum, digest, lock
from .scene_store import ROW_DTYPE, row_digest
from .physical_sampling import perturb, physical_mask, proposal_parameters, score_mask, select_bank


def path_key(path):
    a = np.array(path, dtype="<f4", copy=True)
    a[a == 0] = 0
    return a.tobytes()


def validate_physical_rows(rows, constructor):
    from .filter_targets import TargetPolicy

    p = TargetPolicy(**constructor["policy"])
    mask, _ = physical_mask(
        rows["trajectories"],
        np.asarray(constructor["expert"]),
        p,
        constructor["initial_velocity"],
        constructor["initial_acceleration"],
    )
    y = rows["labels"]
    if not mask.all() or not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError("Published physical constraints or score range failed")
    expected = y[:, 0] * y[:, 1] * (5 * y[:, 2] + 5 * y[:, 3] + 2 * y[:, 4]) / 12
    if not np.allclose(expected, y[:, -1], rtol=0, atol=1e-9):
        raise ValueError("Official total/component mismatch")
    if len({path_key(t) for t in rows["trajectories"]}) != len(rows):
        raise ValueError("Duplicate saved paths")


def generate_scene(record, config):
    import torch

    torch.set_num_threads(1)
    from .simscale_source import load_source, seed_paths
    from .filter_targets import (
        TargetPolicy,
        planned_space_checker,
        tracking_errors,
        MissingPlannedGeometry,
    )

    output = Path(config["output"])
    folder = output / "scenes" / record["split"] / record["token"]
    with lock(folder):
        result_path = folder / "RESULT.json"
        if result_path.exists():
            result = read_json(result_path)
            compatible = (
                result["config_digest"] in config.get("_compatible_digests", [])
                and result.get("complete_high")
                and result.get("complete_full")
            )
            if result["config_digest"] != config["digest"] and not compatible:
                raise ValueError("Scene was generated under a different configuration")
            for name, value in result["files"].items():
                if checksum(folder / name) != value:
                    raise ValueError("Completed scene file corrupted: " + name)
            return dict(result, reused=True)
        begin = time.time()
        cache, expert, identity = load_source(record, output, config.get("actor_tail"))
        seeds, seed_meta, evaluator, origin, velocity, acceleration = seed_paths(cache, expert)
        adaptive = config.get("adaptive_sampling", False)
        if adaptive:
            from .adaptive_sampling import (
                forward_seeds,
                perturb_preserving_heading,
                shuffled_parameters,
                diverse_indices,
                quality,
            )

            added, added_meta = forward_seeds(expert, velocity, acceleration)
            seeds = np.concatenate([seeds, added])
            seed_meta.extend(added_meta)
        policy = TargetPolicy(**config["policy"])
        try:
            from .physical_sampling import safe_planned_space_checker

            spatial_check, omitted_geometry = safe_planned_space_checker(cache)
            if adaptive:
                from .adaptive_sampling import PreparedSpace

                spatial_check = PreparedSpace(cache)
            identity["spatial_screen_excluded_invalid_polygons"] = omitted_geometry
            missing_geometry = False
        except MissingPlannedGeometry:
            spatial_check = lambda t: False
            missing_geometry = True
        initial_space_valid = (
            bool(spatial_check(np.zeros((8, 3), np.float32))) if adaptive else True
        )
        checkpoint = folder / "checkpoint.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as z:
                if z["config_digest"].tobytes().decode("ascii") != config["digest"]:
                    raise ValueError("Checkpoint configuration changed")
                paths, labels, eligible = (
                    list(z["trajectories"]),
                    list(z["labels"]),
                    list(z["eligible"]),
                )
                provenance, track = list(z["provenance"]), list(z["tracking"])
                alternative = list(z["alternative_labels"])
                cursor, proposed, rejected = map(int, z["counters"])
                if adaptive:
                    priorities = list(z["priorities"])
                    resumed_seeds = z["seeds"].copy()
                    resumed_meta = json.loads(z["seed_metadata"].tobytes().decode())
        else:
            paths, labels, eligible, provenance, track, alternative = [], [], [], [], [], []
            cursor = proposed = rejected = 0
            priorities = []
        seen = {path_key(t) for t in paths}
        # Raw GT is an independent expert target; do not count unchanged GT as new data.
        seen.add(path_key(expert))
        seen.add(path_key(np.asarray(identity["raw_expert_gt"], np.float32)))
        seed_scores, seed_sim = evaluator.score(seeds)
        seed_alternative = evaluator.alternative_labels.copy()
        seed_physics, _ = physical_mask(seeds, expert, policy, velocity, acceleration)
        seed_order = np.argsort(-seed_scores[:, -1], kind="stable")
        valid_seed_ids = [int(i) for i in seed_order if seed_physics[i]]
        # Even a seed that fails the precheck can generate a valid small correction.
        parent_ids = (
            valid_seed_ids + [int(i) for i in seed_order if int(i) not in valid_seed_ids]
        )[:12]
        if adaptive:
            parent_ids = list(range(len(seeds)))
        seed_eligible = []
        seed_errors, seed_space = [], []
        for i in parent_ids:
            errs = tracking_errors(seeds[i], seed_sim[i], origin)
            space = bool(spatial_check(seeds[i]))
            seed_errors.append(errs)
            seed_space.append(space)
            seed_eligible.append(
                bool(
                    seed_physics[i]
                    and score_mask(seed_scores[i : i + 1], policy)[0]
                    and score_mask(seed_alternative[i : i + 1], policy)[0]
                    and space
                    and max(
                        errs[j] / limit
                        for j, limit in enumerate(
                            [
                                policy.max_tracking_ade_m,
                                policy.max_tracking_error_m,
                                policy.max_tracking_heading_error_rad,
                            ]
                        )
                    )
                    <= 1
                )
            )
        preferred = [i for i, ok in zip(parent_ids, seed_eligible) if ok]
        parent_ids = preferred + [i for i in parent_ids if i not in preferred]
        limits = [
            policy.max_tracking_ade_m,
            policy.max_tracking_error_m,
            policy.max_tracking_heading_error_rad,
        ]
        if adaptive:
            seed_quality = quality(
                seed_scores, seed_alternative, np.asarray(seed_errors), limits, seed_space
            )
            seed_quality -= 0.5 * (~seed_physics)
            ranked = diverse_indices(seeds, seed_quality, 24, 0.05)
            parent_ids = preferred + [i for i in ranked if i not in preferred]
            # Tracking projections are proposals, never copied labels.
            from .recovery_seeds import to_local

            for i in ranked[:8]:
                projected = to_local(seed_sim[i, :, :3], origin)[5::5]
                if physical_mask(projected[None], expert, policy, velocity, acceleration)[0][
                    0
                ] and not any(np.array_equal(projected, s) for s in seeds):
                    parent_ids.append(len(seeds))
                    seeds = np.concatenate([seeds, projected[None]])
                    seed_meta.append(dict(parent="initial_tracking_projection", source_seed=i))
            if checkpoint.exists():
                if not np.array_equal(resumed_seeds[: len(seeds)], seeds):
                    raise ValueError("Adaptive seed construction changed during resume")
                seeds, seed_meta = resumed_seeds, resumed_meta
        seed = int(hashlib.sha256(record["token"].encode()).hexdigest()[:8], 16)
        params = (
            shuffled_parameters(config["max_proposals"], seed)
            if adaptive
            else proposal_parameters(config["max_proposals"], seed)
        )
        batch_size = config["score_batch"]
        audit_queries = 0

        def save_checkpoint():
            extra_checkpoint = (
                dict(
                    priorities=np.asarray(priorities),
                    seeds=seeds,
                    seed_metadata=np.frombuffer(json.dumps(seed_meta).encode(), dtype=np.uint8),
                )
                if adaptive
                else {}
            )
            write_npz(
                checkpoint,
                trajectories=np.asarray(paths, np.float32).reshape(-1, 8, 3),
                labels=np.asarray(labels, float).reshape(-1, 7),
                eligible=np.asarray(eligible, bool),
                alternative_labels=np.asarray(alternative, float).reshape(-1, 7),
                provenance=np.asarray(provenance, float).reshape(-1, 8),
                tracking=np.asarray(track, float).reshape(-1, 3),
                counters=np.array([cursor, proposed, rejected]),
                config_digest=np.frombuffer(config["digest"].encode("ascii"), dtype=np.uint8),
                **extra_checkpoint,
            )

        while cursor < config["max_proposals"] and len(paths) < config["max_queries"]:
            if len(paths) >= config["full_count"] and sum(eligible) >= config["high_count"]:
                break
            if adaptive and not initial_space_valid and len(paths) >= config["full_count"]:
                break
            stop = min(cursor + 512, config["max_proposals"])
            indices = np.arange(cursor, stop)
            # Mix wide support and local refinements; physically invalid proposals
            # are rejected before any tracking simulation or score computation.
            candidate_paths, candidate_prov = None, []
            local_ids = preferred.copy()
            qualified_count = int(np.count_nonzero(eligible))
            refine_high = qualified_count < config["high_count"] and bool(
                preferred or qualified_count
            )
            if adaptive and paths:
                available = np.flatnonzero(eligible)
                if len(available):
                    chosen = diverse_indices(
                        np.asarray(paths)[available], np.asarray(labels)[available, -1], 16, 0.025
                    )
                    chosen = available[chosen]
                else:
                    chosen = np.asarray(diverse_indices(np.asarray(paths), priorities, 12, 0.1))
                for k in chosen:
                    matches = [i for i, s in enumerate(seeds) if np.array_equal(s, paths[k])]
                    if matches:
                        local_ids.append(matches[0])
                    else:
                        local_ids.append(len(seeds))
                        seeds = np.concatenate([seeds, np.asarray(paths[k])[None]])
                        seed_meta.append(
                            dict(
                                parent="scored_query",
                                source_query=int(k),
                                qualified=bool(eligible[k]),
                            )
                        )
                local_ids = list(dict.fromkeys(local_ids))
            for j in indices:
                parent = parent_ids[(j // 4) % len(parent_ids)]
                # Once high-quality references exist, keep exploring every fourth proposal.
                if preferred and j % 4 != 0:
                    parent = preferred[(j // 4) % len(preferred)]
                scale = (0.1, 0.35, 1.0, 2.0)[j % 4]
                if adaptive and j % 4 == 0:
                    scale = (0.1, 0.35, 1.0, 2.0)[(j // 4) % 4]
                if adaptive and local_ids and j % 4 != 0:
                    parent = local_ids[(j // 4) % len(local_ids)]
                    if refine_high:
                        scale = (0.025, 0.05, 0.1)[j % 4 - 1]
                    else:
                        scale = (0.1, 0.35, 1.0)[j % 4 - 1]
                coeff = np.round(params[j] * np.array([3.0, 1.0, 1.0, 0.5, 0.5, 0.25]) * scale, 3)
                candidate_prov.append([parent, j, *coeff])
            candidate_prov = np.asarray(candidate_prov)
            for parent in np.unique(candidate_prov[:, 0]).astype(int):
                ids = np.flatnonzero(candidate_prov[:, 0] == parent)
                if candidate_paths is None:
                    candidate_paths = np.zeros((len(indices), 8, 3), np.float32)
                candidate_paths[ids] = (perturb_preserving_heading if adaptive else perturb)(
                    seeds[parent], candidate_prov[ids, 2:]
                )
            physical, _ = physical_mask(candidate_paths, expert, policy, velocity, acceleration)
            proposed += len(indices)
            rejected += int((~physical).sum())
            keep = []
            for k in np.flatnonzero(physical):
                key = path_key(candidate_paths[k])
                if key not in seen:
                    keep.append(k)
                    seen.add(key)
            # Finish this deterministic proposal block before advancing cursor.
            # Checkpoints always contain complete blocks, so interrupted scoring
            # can repeat only uncommitted work, never skip a candidate.
            keep = keep[: config["max_queries"] - len(paths)]
            for offset in range(0, len(keep), batch_size):
                ids = keep[offset : offset + batch_size]
                new_paths = candidate_paths[ids]
                ys, sims = evaluator.score(new_paths)
                alt = evaluator.alternative_labels.copy()
                errors = np.array(
                    [tracking_errors(t, sim, origin) for t, sim in zip(new_paths, sims)]
                )
                ok = score_mask(ys, policy)
                ok &= score_mask(alt, policy)
                ok &= (
                    errors
                    <= [
                        policy.max_tracking_ade_m,
                        policy.max_tracking_error_m,
                        policy.max_tracking_heading_error_rad,
                    ]
                ).all(1)
                if adaptive:
                    spatial = (
                        spatial_check.batch(new_paths)
                        if not missing_geometry
                        else np.zeros(len(new_paths), bool)
                    )
                    ok &= spatial
                    priorities.extend(quality(ys, alt, errors, limits, spatial))
                else:
                    for k in np.flatnonzero(ok):
                        ok[k] = spatial_check(new_paths[k])
                # Singleton witness includes a high and a low candidate where available.
                if not paths:
                    for k in set([0, int(np.argmax(ys[:, -1])), int(np.argmin(ys[:, -1]))]):
                        single = evaluator.runtime.score(new_paths[k])
                        audit_queries += 1
                        if not np.allclose(single, ys[k], atol=1e-8, rtol=0):
                            raise ValueError("Batched labels differ from official singleton")
                paths.extend(new_paths)
                labels.extend(ys)
                eligible.extend(ok)
                alternative.extend(alt)
                provenance.extend(candidate_prov[ids])
                track.extend(errors)
            cursor = stop
            save_checkpoint()
        paths = np.asarray(paths, np.float32).reshape(-1, 8, 3)
        labels = np.asarray(labels, float).reshape(-1, 7)
        eligible = np.asarray(eligible, bool)
        full_ids, high_ids = select_bank(
            paths, labels, eligible, config["full_count"], config["high_count"]
        )
        full = np.zeros(len(full_ids), dtype=ROW_DTYPE)
        full["keys"][:, 1] = full_ids
        full["trajectories"], full["labels"] = paths[full_ids], labels[full_ids]
        constructor = dict(
            format="physical_simscale_v1",
            policy=asdict(policy),
            expert=expert.tolist(),
            initial_velocity=velocity,
            initial_acceleration=acceleration,
            key_semantics=["reserved_zero", "source_query_index"],
            config_digest=config["digest"],
        )
        validate_physical_rows(full, constructor)
        contract = dict(
            identity=identity,
            protocol={
                "constructor": constructor,
                "sampling": {"mode": "bounded_reference_sobol"},
                "scoring_protocol": config["scoring_protocol"],
                "high_selection": "nominal_and_alternative_pdm95_plus_driving_checks",
            },
        )
        write_json(folder / "contract.json", contract)
        write_json(
            folder / "expert.json",
            dict(
                trajectory=expert.tolist(),
                raw_trajectory=identity["raw_expert_gt"],
                labels=evaluator.runtime.score(expert).tolist(),
            ),
        )
        atomic(folder / "candidates.npy", lambda f: np.save(f, full, allow_pickle=False))
        write_npz(
            folder / "construction.npz",
            seeds=seeds,
            parameters=np.asarray(provenance, float)[full_ids],
            tracking=np.asarray(track, float)[full_ids],
            eligible=eligible[full_ids],
            alternative_labels=np.asarray(alternative, float).reshape(-1, 7)[full_ids],
        )
        write_json(
            folder / "seeds.json",
            dict(
                metadata=seed_meta,
                labels=seed_scores.tolist(),
                scored_seed_count=len(seed_scores),
                physics_pass=seed_physics.tolist(),
                selected_parents=parent_ids,
            ),
        )
        extra = np.setdiff1d(np.arange(len(paths)), full_ids)
        if len(extra):
            write_npz(
                folder / "extra_queries.npz",
                trajectories=paths[extra],
                labels=labels[extra],
                eligible=eligible[extra],
                source_query_indices=extra,
                parameters=np.asarray(provenance, float)[extra],
                alternative_labels=np.asarray(alternative, float).reshape(-1, 7)[extra],
                tracking=np.asarray(track, float)[extra],
            )
        write_json(
            folder / "state.json",
            dict(
                count=len(full),
                pending=None,
                completed_stages=["physical_search"],
                blocks=[dict(start=0, end=len(full), digest=row_digest(full))] if len(full) else [],
            ),
        )
        complete_full = len(full) == config["full_count"]
        if complete_full:
            write_json(
                folder / "ALL_SCORES.json",
                dict(
                    format="all_scored_candidates_v1",
                    passed=True,
                    candidates=len(full),
                    token=record["token"],
                    split=record["split"],
                    data_digest=checksum(folder / "candidates.npy"),
                ),
            )
        complete_high = complete_full and len(high_ids) == config["high_count"]
        official_error = 0.0
        if complete_high:
            # Every high target is re-simulated in a fresh evaluator and a
            # different batch partition. Singleton witnesses test equivalence.
            from .batched_official import BatchedOfficial

            independent = BatchedOfficial(cache)
            for start in range(0, len(high_ids), 31):
                ids = high_ids[start : start + 31]
                checked, _ = independent.score(full["trajectories"][ids])
                audit_queries += len(ids)
                official_error = max(
                    official_error, float(abs(checked - full["labels"][ids]).max())
                )
                expected_alt = np.asarray(alternative, float)[full_ids[ids]]
                official_error = max(
                    official_error, float(abs(independent.alternative_labels - expected_alt).max())
                )
                if not score_mask(independent.alternative_labels, policy).all():
                    raise ValueError("Alternative actor tail rejects published high targets")
            witness_count = min(int(config.get("audit_singletons", 3)), len(high_ids))
            for i in high_ids[np.linspace(0, len(high_ids) - 1, witness_count, dtype=int)]:
                single = evaluator.runtime.score(full["trajectories"][i])
                audit_queries += 1
                official_error = max(official_error, float(abs(single - full["labels"][i]).max()))
            if official_error > 1e-8:
                raise ValueError("Selected 256 failed independent official verification")
            write_npz(
                folder / "high_score.npz",
                **{k: full[k][high_ids] for k in ("keys", "trajectories", "labels")},
                query_indices=high_ids,
            )
            from .high_score_bank import validate_bank

            with np.load(folder / "high_score.npz", allow_pickle=False) as z:
                validate_bank(z, expert, contract["protocol"])
            write_json(
                folder / "COMPLETE.json",
                dict(
                    format="high_score_256",
                    passed=True,
                    candidates=256,
                    token=record["token"],
                    split=record["split"],
                    data_digest=checksum(folder / "high_score.npz"),
                    official_max_error=official_error,
                    minimum_pdms=float(full["labels"][high_ids, -1].min() * 100),
                ),
            )
        else:
            write_json(
                folder / "SHORTFALL.json",
                dict(
                    token=record["token"],
                    full_count=len(full),
                    high_count=len(high_ids),
                    qualified_queries=int(eligible.sum()),
                    best_pdms=float(labels[:, -1].max() * 100) if len(labels) else None,
                    missing_geometry=missing_geometry,
                    proposal_limit=cursor == config["max_proposals"],
                    query_limit=len(paths) == config["max_queries"],
                    initial_space_valid=initial_space_valid,
                ),
            )
        result = dict(
            token=record["token"],
            source=record["source"],
            config_digest=config["digest"],
            full_count=len(full),
            high_count=len(high_ids),
            complete_full=complete_full,
            complete_high=complete_high,
            scored_queries=len(paths),
            audit_queries=audit_queries,
            proposed=proposed,
            rejected_physical=rejected,
            actor_tail_actors=identity.get("actor_tail", {}).get("actor_count", 0),
            initial_space_valid=initial_space_valid,
            tail_sensitive_queries=int(
                np.count_nonzero(
                    np.asarray(alternative, float).reshape(-1, 7)[:, 3] != labels[:, 3]
                )
            ),
            elapsed_s=time.time() - begin,
            files={},
        )
        files = [
            "candidates.npy",
            "construction.npz",
            "contract.json",
            "expert.json",
            "state.json",
            "seeds.json",
        ]
        files += ["ALL_SCORES.json"] if complete_full else []
        files += ["COMPLETE.json", "high_score.npz"] if complete_high else ["SHORTFALL.json"]
        files += ["extra_queries.npz"] if len(extra) else []
        result["files"] = {name: checksum(folder / name) for name in files}
        write_json(result_path, result)
        checkpoint.unlink(missing_ok=True)
        return result


def worker(record, config):
    try:
        return generate_scene(record, config)
    except Exception:
        result = dict(token=record["token"], source=record["source"], error=traceback.format_exc())
        write_json(Path(config["output"]) / "errors" / (record["token"] + ".json"), result)
        return result


def run(config, records, workers, compatible_digests=()):
    config = dict(config, _compatible_digests=list(compatible_digests))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / "runner"):
        results = []
        start = time.time()
        last = 0

        def report(final=False):
            good = [x for x in results if "error" not in x]
            errors = [x["token"] for x in results if "error" in x]
            missing = [x["token"] for x in good if not x["complete_high"]]
            elapsed = time.time() - start
            reused = sum(x.get("reused", False) for x in good)
            generated = sum(not x.get("reused", False) for x in good)
            summary = dict(
                status=(
                    "complete"
                    if final and not errors and not missing
                    else "partial" if final else "running"
                ),
                pid=os.getpid(),
                workers=workers,
                scenes=len(records),
                processed=len(results),
                errors=len(errors),
                full3072=sum(x["complete_full"] for x in good),
                high256=sum(x["complete_high"] for x in good),
                shortfall=len(missing),
                scored_queries=sum(x["scored_queries"] for x in good),
                elapsed_s=elapsed,
                reused=reused,
                newly_generated=generated,
                scene_per_hour=generated / max(elapsed, 1) * 3600,
                updated_unix=time.time(),
                eta_hours=(
                    (len(records) - len(results)) * elapsed / generated / 3600
                    if generated
                    else None
                ),
            )
            write_json(output / "status.json", summary)
            atomic(
                output / "shortfall_tokens.txt",
                lambda f: f.write(("\n".join(missing) + "\n").encode()),
            )
            atomic(
                output / "error_tokens.txt", lambda f: f.write(("\n".join(errors) + "\n").encode())
            )
            print(json.dumps(summary), flush=True)
            if final:
                write_json(
                    output
                    / ("COMPLETE.json" if summary["status"] == "complete" else "PARTIAL.json"),
                    dict(summary, passed=summary["status"] == "complete"),
                )

        with (
            ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool,
            (output / "results.jsonl").open("a") as stream,
        ):
            iterator = iter(records)
            pending = {}

            def fill():
                while len(pending) < workers * 2:
                    try:
                        r = next(iterator)
                    except StopIteration:
                        break
                    pending[pool.submit(worker, r, config)] = r["token"]

            fill()
            report()
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    token = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        result = dict(token=token, error=traceback.format_exc())
                    results.append(result)
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                fill()
                if time.time() - last >= 30:
                    report()
                    last = time.time()
            report(True)


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--tokens", type=Path, help="Fixed scene-token subset for paired validation"
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Archive prior shortfalls and preserve compatible completed banks",
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--audit-singletons", type=int)
    args = parser.parse_args()
    from drivesta_feedback.paths import bootstrap_navsim

    root = bootstrap_navsim()
    cfg = yaml.safe_load(args.config.read_text())
    cfg["output"] = str(args.output or root / cfg["output"])
    if cfg.get("scoring_protocol") != "navsim_v1_full_ttc_actor_tail_v1" or "actor_tail" not in cfg:
        raise ValueError("This runner requires the explicit inferred-actor full TTC protocol")
    if args.audit_singletons is not None:
        cfg["audit_singletons"] = args.audit_singletons
    cfg["implementation"] = {
        p.name: checksum(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("simscale_source.py"),
            Path(__file__).with_name("physical_sampling.py"),
            Path(__file__).with_name("batched_official.py"),
            Path(__file__).with_name("filter_targets.py"),
            Path(__file__).with_name("actor_tail.py"),
            Path(__file__).with_name("adaptive_sampling.py"),
        ]
    }
    cfg["digest"] = digest({k: v for k, v in cfg.items() if k != "output"})
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    from .simscale_source import manifest

    if (output / "run.json").exists():
        previous = read_json(output / "run.json")
        if previous["config"] != cfg and args.upgrade:
            from .simscale_upgrade import upgrade_run

            previous = upgrade_run(output, cfg, args.limit)
        if previous["config"] != cfg or previous["limit"] != args.limit:
            raise ValueError("Run contract changed")
        records = [json.loads(s) for s in (output / "scenes.jsonl").read_text().splitlines()]
    else:
        records, counts = manifest(root, cfg)
        if args.tokens:
            tokens = set(args.tokens.read_text().split())
            records = [r for r in records if r["token"] in tokens]
            if len(records) != len(tokens):
                raise ValueError("Requested token missing from source manifest")
        if args.limit:
            rng = np.random.default_rng(20261020)
            ids = sorted(rng.choice(len(records), args.limit, replace=False))
            records = [records[i] for i in ids]
        atomic(
            output / "scenes.jsonl",
            lambda f: f.write(("".join(json.dumps(r) + "\n" for r in records)).encode()),
        )
        write_json(
            output / "run.json",
            dict(config=cfg, limit=args.limit, source_counts=counts, scenes=len(records)),
        )
    if not args.manifest_only:
        # This runner currently uses NAVSIM-v1 full-window labels. SimScale's
        # within-window TTC needs a separate explicit scoring protocol.
        # Test one record per source before dispatching any production jobs.
        import lzma, pickle
        from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

        install_numpy_2_pickle_compat()
        import numpy.core.numeric, sys

        sys.modules["numpy._core.numeric"] = numpy.core.numeric
        from .simscale_source import require_observation_horizon

        checked = {}
        for r in records:
            if r["source"] in checked:
                continue
            with lzma.open(r["source_cache_path"], "rb") as f:
                c = pickle.load(f)
            try:
                require_observation_horizon(c, 4.0)
                from .actor_tail import extend_observation, TailPolicy
                import copy

                c = copy.copy(c)
                c.observation, _, meta = extend_observation(
                    c.observation, policy=TailPolicy(**cfg["actor_tail"])
                )
                require_observation_horizon(c)
                checked[r["source"]] = dict(
                    passed=True, token=r["token"], actor_count=meta["actor_count"]
                )
            except ValueError as e:
                checked[r["source"]] = dict(passed=False, token=r["token"], reason=str(e))
        write_json(output / "source_preflight.json", checked)
        if not all(v["passed"] for v in checked.values()):
            raise ValueError(
                "Source observation horizon is incomplete; see source_preflight.json. Full generation was not launched."
            )
        run(
            cfg, records, args.workers, read_json(output / "run.json").get("compatible_digests", [])
        )


if __name__ == "__main__":
    main()
