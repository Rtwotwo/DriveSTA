"""Bounded recovery around verified seeds; original banks remain read only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np


def grid_keys():
    """Coarse neighborhood followed by finer coverage, without repeated goals."""
    keys, seen = [], set()
    for radius, step in ((0.6, 0.1), (0.4, 0.025), (0.8, 0.05), (0.25, 0.01)):
        values = np.arange(-round(radius / step), round(radius / step) + 1) * round(step * 1000)
        points = sorted(
            ((int(x), int(y)) for x in values for y in values),
            key=lambda p: (p[0] ** 2 + p[1] ** 2, p),
        )
        for key in points:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return np.asarray(keys, np.int32)


def interior_paths(seed, keys):
    """Change interior geometry while preserving first and terminal poses."""
    seed = np.asarray(seed, float)
    u = np.linspace(0, 1, 8)
    bump = 16 * u * u * (1 - u) * (1 - u)
    result = np.broadcast_to(seed, (len(keys), 8, 3)).copy()
    result[:, :, :2] += bump[None, :, None] * np.asarray(keys)[:, None, :] * 0.001
    base_delta = np.gradient(seed[:, :2], axis=0)
    delta = np.gradient(result[:, :, :2], axis=1)
    change = np.arctan2(delta[..., 1], delta[..., 0]) - np.arctan2(
        base_delta[:, 1], base_delta[:, 0]
    )
    angle = seed[None, :, 2] + np.arctan2(np.sin(change), np.cos(change))
    result[:, :, 2] = np.arctan2(np.sin(angle), np.cos(angle))
    result[:, [0, -1]] = seed[[0, -1]]
    result[np.all(np.asarray(keys) == 0, axis=1)] = seed
    return result.astype(np.float32)


def paired_interior_keys(radii_mm=(50, 100, 200)):
    """Bounded four-dimensional geometry queries on a millimetre lattice."""
    from scipy.stats import qmc

    samples = 2 * qmc.Sobol(d=4, scramble=True, seed=548).random_base2(10) - 1
    keys, seen = [(0, 0, 0, 0)], {(0, 0, 0, 0)}
    for radius_mm in radii_mm:
        for row in np.round(samples * radius_mm).astype(np.int32):
            key = tuple(int(x) for x in row)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return np.asarray(keys, np.int32)


def paired_interior_paths(seed, keys):
    """Independent early/late interior changes with exact endpoint preservation."""
    seed = np.asarray(seed, float)
    keys = np.asarray(keys, float) * 0.001
    u = np.linspace(0, 1, 8)
    first = 16 * u * u * (1 - u) * (1 - u)
    second = first * (2 * u - 1)
    result = np.broadcast_to(seed, (len(keys), 8, 3)).copy()
    result[:, :, :2] += (
        first[None, :, None] * keys[:, None, :2] + second[None, :, None] * keys[:, None, 2:]
    )
    base = np.gradient(seed[:, :2], axis=0)
    delta = np.gradient(result[:, :, :2], axis=1)
    change = np.arctan2(delta[..., 1], delta[..., 0]) - np.arctan2(base[:, 1], base[:, 0])
    angle = seed[None, :, 2] + np.arctan2(np.sin(change), np.cos(change))
    result[:, :, 2] = np.arctan2(np.sin(angle), np.cos(angle))
    result[:, [0, -1]] = seed[[0, -1]]
    result[np.all(keys == 0, axis=1)] = seed
    return result.astype(np.float32)


def choose_cohort(seed_root, limit, seed_rank=0, tokens=None):
    preferred = (
        "timestamp_expert",
        "tracked_pdm_teacher",
        "offline_pdm_teacher",
        "expert_teacher_blend",
        "teacher_time_0.95",
        "teacher_time_0.90",
        "teacher_time_0.85",
    )
    groups = {name: [] for name in preferred}
    for path in sorted((Path(seed_root) / "scenes").glob("*.json")):
        row = json.loads(path.read_text())
        if tokens is not None and row["record"]["token"] not in tokens:
            continue
        if seed_rank:
            eligible = sorted(
                (k for k, a in row["arms"].items() if a["generator_eligible"]),
                key=lambda k: -row["arms"][k]["pdms"],
            )
            if len(eligible) > seed_rank:
                name = eligible[seed_rank]
                groups.setdefault(name, []).append(dict(record=row["record"], seed=name))
            continue
        order = list(preferred) + sorted(
            set(row["arms"]) - set(preferred), key=lambda k: -row["arms"][k]["pdms"]
        )
        for name in order:
            if name in row["arms"] and row["arms"][name]["generator_eligible"]:
                groups.setdefault(name, []).append(dict(record=row["record"], seed=name))
                break
    chosen = []
    # Spread selections over each source group; these are feasibility cases,
    # not an unbiased estimate of recovery across the full 548 scenes.
    order = {
        k: [v[i] for i in np.linspace(0, len(v) - 1, min(len(v), limit)).round().astype(int)]
        for k, v in groups.items()
        if v
    }
    while len(chosen) < limit and any(order.values()):
        for rows in order.values():
            if rows and len(chosen) < limit:
                chosen.append(rows.pop(0))
    return chosen, {k: len(v) for k, v in groups.items()}


def recover_one(job):
    item, config = job
    import torch
    from .source import load_scene, Evaluator, construct
    from .gt_neighborhood import relative_heading
    from .filter_targets import (
        TargetPolicy,
        preliminary_rules,
        planned_space_checker,
        tracking_errors,
    )
    from .store import read_json, write_json, atomic

    torch.set_num_threads(1)
    record = item["record"]
    folder = Path(config["source"]) / "scenes" / record["split"] / record["token"]
    output = Path(config["output"]) / "scenes" / record["token"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        return read_json(output / "result.json")
    contract = read_json(folder / "contract.json")
    scene, expert, cache, identity = load_scene(record, [record["token"]], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source observation/cache differs")
    with np.load(
        Path(config["seeds"]) / "scenes" / (record["token"] + ".npz"), allow_pickle=False
    ) as z:
        seed = z["trajectories"][list(z["names"]).index(item["seed"])]
    protocol = dict(
        sampling={"integer_lattice_m": 0.001}, constructor={"max_planned_endpoint_error_m": 3e-5}
    )
    policy = TargetPolicy(**config["policy"])
    evaluator = Evaluator(cache)
    checker = planned_space_checker(cache)
    pose = cache.ego_state.rear_axle
    origin = [pose.x, pose.y, pose.heading]
    dynamic = cache.ego_state.dynamic_car_state
    velocity = [dynamic.rear_axle_velocity_2d.x, dynamic.rear_axle_velocity_2d.y]
    acceleration = [dynamic.rear_axle_acceleration_2d.x, dynamic.rear_axle_acceleration_2d.y]
    zero = construct(seed, np.zeros((64, 2), np.int32), protocol)[0][0]
    operator = config.get("operator", "endpoint")
    if operator == "paired_interior_fine":
        keys = paired_interior_keys((10, 25, 50))
    elif operator == "paired_interior":
        keys = paired_interior_keys()
    else:
        keys = grid_keys()
    keys = keys[: config["max_queries"]]
    paths, labels, accepted, used_keys = [], [], [], []
    seen = {np.asarray(expert, dtype="<f4").tobytes()}
    reasons = {}
    for offset in range(0, len(keys), 64):
        batch = keys[offset : offset + 64]
        padded = np.concatenate([batch, np.repeat(batch[-1:], 64 - len(batch), axis=0)])
        if operator in ("paired_interior", "paired_interior_fine"):
            candidates = paired_interior_paths(seed, batch)
        elif config.get("operator", "endpoint") == "interior":
            candidates = interior_paths(seed, batch)
        else:
            candidates = relative_heading(construct(seed, padded, protocol)[0], zero, seed, padded)[
                : len(batch)
            ]
        for key, path in zip(batch, candidates):
            path_key = np.asarray(path, dtype="<f4").tobytes()
            if path_key in seen:
                continue
            seen.add(path_key)
            label, sim = evaluator.score(path, details=True)
            masks = preliminary_rules(
                path[None], label[None], expert, policy, velocity, acceleration
            )
            failed = [name for name, passed in masks.items() if not passed[0]]
            if not failed:
                if not checker(path):
                    failed.append("planned_space")
                errors = tracking_errors(path, sim, origin)
                if any(
                    a > b
                    for a, b in zip(
                        errors,
                        [
                            policy.max_tracking_ade_m,
                            policy.max_tracking_error_m,
                            policy.max_tracking_heading_error_rad,
                        ],
                    )
                ):
                    failed.append("tracking")
            for name in failed:
                reasons[name] = reasons.get(name, 0) + 1
            paths.append(path)
            labels.append(label)
            used_keys.append(key)
            accepted.append(not failed)
        if sum(accepted) >= 256:
            break
    paths, labels, accepted = np.asarray(paths), np.asarray(labels), np.asarray(accepted, bool)
    candidates = np.flatnonzero(accepted)
    ids = candidates[np.argsort(-labels[candidates, -1], kind="stable")[:256]]
    audit_error = 0.0
    if len(ids) == 256:
        audit = Evaluator(cache)
        checked = np.stack([audit.score(paths[i]) for i in ids])
        audit_error = float(abs(checked - labels[ids]).max())
        if audit_error > 1e-9 or (checked[:, -1] < 0.95).any():
            raise ValueError("Fresh official label audit failed")
    atomic(
        output / "scored.npz",
        lambda f: np.savez(
            f,
            trajectories=paths,
            labels=labels,
            seed_offset_keys=np.asarray(used_keys, np.int32),
            generator_eligible=accepted,
            selected_indices=ids,
            seed=seed,
            operator=config.get("operator", "endpoint"),
        ),
    )
    # Prepare the required fixed-size readout without modifying the live source
    # or mislabelling new paths as old expert-endpoint warps.
    if len(ids) == 256:
        old = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
        chosen_paths, chosen_labels = list(paths[ids]), list(labels[ids])
        selected_seen = {p.tobytes() for p in paths[ids]}
        for i in range(len(old)):
            path = old["trajectories"][i]
            if path.tobytes() not in selected_seen:
                chosen_paths.append(path.copy())
                chosen_labels.append(old["labels"][i].copy())
                selected_seen.add(path.tobytes())
            if len(chosen_paths) == 3072:
                break
        if len(chosen_paths) != 3072:
            raise ValueError("Could not assemble 3072 distinct scored paths")
        atomic(
            output / "training_arrays.npz",
            lambda f: np.savez(
                f,
                trajectories=np.asarray(chosen_paths),
                labels=np.asarray(chosen_labels),
                high_indices=np.arange(256, dtype=np.int64),
                generator_eligible_high_indices=np.arange(256, dtype=np.int64),
            ),
        )
        with np.load(output / "training_arrays.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["trajectories"][:256], paths[ids])
            np.testing.assert_array_equal(z["labels"][:256], labels[ids])
            if len(z["trajectories"]) != 3072:
                raise ValueError("Training arrays readback failed")
    representatives = []
    for i in ids:
        if (
            not representatives
            or np.linalg.norm(paths[representatives, :, :2] - paths[i, :, :2], axis=-1)
            .mean(-1)
            .min()
            >= 0.25
        ):
            representatives.append(i)
    result = dict(
        token=record["token"],
        record=record,
        seed=item["seed"],
        queries=len(paths),
        operator=config.get("operator", "endpoint"),
        selected=len(ids),
        eligible=int(accepted.sum()),
        passed=len(ids) == 256,
        failure_counts=reasons,
        best_pdms=float(labels[:, -1].max() * 100),
        minimum_selected_pdms=float(labels[ids, -1].min() * 100) if len(ids) else None,
        distinct_representatives_025m=len(representatives),
        max_official_label_error=audit_error,
        published_to_live_bank=False,
        training_array_count=3072 if len(ids) == 256 else 0,
    )
    write_json(output / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--seeds", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-queries", type=int, default=1024)
    p.add_argument("--seed-rank", type=int, default=0)
    p.add_argument(
        "--operator",
        choices=("endpoint", "interior", "paired_interior", "paired_interior_fine"),
        default="endpoint",
    )
    p.add_argument("--tokens", type=Path)
    args = p.parse_args()
    import yaml
    from .store import write_json, read_json

    if args.seed_rank < 0:
        raise ValueError("Seed rank must be nonnegative")
    selected_tokens = args.tokens.read_text().split() if args.tokens else None
    if selected_tokens is not None:
        if len(selected_tokens) != len(set(selected_tokens)):
            raise ValueError("Repeated input token")
        if any(not (args.seeds / "scenes" / (t + ".json")).exists() for t in selected_tokens):
            raise ValueError("Input token has no seed record")
    cohort, groups = choose_cohort(args.seeds, args.count, args.seed_rank, selected_tokens)
    config = dict(
        version=1,
        source=str(args.source),
        seeds=str(args.seeds),
        output=str(args.output),
        policy=yaml.safe_load(args.policy.read_text()),
        max_queries=args.max_queries,
        cohort=cohort,
    )
    if args.operator != "endpoint":
        config["operator"] = args.operator
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists() and read_json(args.output / "config.json") != config:
        raise ValueError("Pilot configuration changed")
    write_json(args.output / "config.json", config)
    results, errors, start = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        jobs = {
            pool.submit(recover_one, (item, config)): item["record"]["token"] for item in cohort
        }
        for future in as_completed(jobs):
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
            except Exception:
                error = dict(token=jobs[future], traceback=traceback.format_exc())
                errors.append(error)
                print(json.dumps(error), flush=True)
    summary = dict(
        scenes=len(results),
        passed=sum(r["passed"] for r in results),
        errors=errors,
        eligible_seed_groups=groups,
        elapsed_s=time.time() - start,
        queries=sum(r["queries"] for r in results),
        recovered_tokens=[r["token"] for r in results if r["passed"]],
        results=results,
        published_to_live_bank=False,
    )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("Recovery pilot contains errors")


if __name__ == "__main__":
    main()
