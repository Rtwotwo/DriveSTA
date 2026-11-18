"""Inspect time-aligned expert and offline PDM-teacher seeds for shortfalls.

The official evaluator and source banks are read only. No seed inherits a score.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np


def resample_recorded(poses, times, desired=None):
    desired = np.arange(1, 9) * 0.5 if desired is None else np.asarray(desired)
    poses, times = np.asarray(poses, float), np.asarray(times, float)
    if (
        poses.shape != (len(times), 3)
        or not np.isfinite(poses).all()
        or not np.isfinite(times).all()
    ):
        raise ValueError("Invalid recorded poses/times")
    if np.any(np.diff(times) <= 0) or desired.min() < times[0] or desired.max() > times[-1]:
        raise ValueError("Observed future does not cover requested times; extrapolation disabled")
    result = np.column_stack(
        [np.interp(desired, times, poses[:, k]) for k in (0, 1)]
        + [np.interp(desired, times, np.unwrap(poses[:, 2]))]
    )
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result.astype(np.float32)


def to_local(poses, origin):
    poses, origin = np.asarray(poses, float), np.asarray(origin, float)
    c, s = np.cos(origin[2]), np.sin(origin[2])
    result = poses.copy()
    result[:, :2] = (poses[:, :2] - origin[:2]) @ np.array([[c, -s], [s, c]])
    result[:, 2] = np.arctan2(np.sin(poses[:, 2] - origin[2]), np.cos(poses[:, 2] - origin[2]))
    return result.astype(np.float32)


def inspect_scene(job):
    record, source, policy_values, output = job
    import torch
    from nuplan.common.actor_state.state_representation import TimePoint
    from .source import load_scene, Evaluator
    from .filter_targets import (
        TargetPolicy,
        preliminary_rules,
        planned_space_checker,
        tracking_errors,
        MissingPlannedGeometry,
    )
    from .store import read_json, write_json, atomic

    torch.set_num_threads(1)
    directory = Path(source) / "scenes" / record["split"] / record["token"]
    contract = read_json(directory / "contract.json")
    scene, gt, cache, identity = load_scene(record, [record["token"]], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source identity mismatch")
    original = np.load(directory / "candidates.npy", mmap_mode="r", allow_pickle=False)
    if len(original) != 3072 or not np.isfinite(original["labels"]).all():
        raise ValueError("Missing/invalid original full3072 table")
    start = scene.scene_metadata.num_history_frames - 1
    times = (
        np.array([f.timestamp for f in scene.frames[start:]], float) - scene.frames[start].timestamp
    ) / 1e6
    future = scene.get_future_trajectory(num_trajectory_frames=len(times) - 1).poses
    timed = resample_recorded(np.vstack((np.zeros((1, 3)), future)), times)
    pose = cache.ego_state.rear_axle
    origin = np.array([pose.x, pose.y, pose.heading])
    requests = [
        TimePoint(cache.ego_state.time_point.time_us + round(t * 1e6))
        for t in np.arange(1, 9) * 0.5
    ]
    if requests[-1].time_us > cache.trajectory.end_time.time_us:
        raise ValueError("Offline PDM teacher does not cover the full four seconds")
    states = cache.trajectory.get_state_at_times(requests)
    reference = to_local(
        np.array([[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading] for s in states]), origin
    )
    blend = (timed + reference) / 2
    delta = np.arctan2(np.sin(reference[:, 2] - timed[:, 2]), np.cos(reference[:, 2] - timed[:, 2]))
    blend[:, 2] = np.arctan2(np.sin(timed[:, 2] + delta / 2), np.cos(timed[:, 2] + delta / 2))
    names = ["raw_expert", "timestamp_expert", "offline_pdm_teacher", "expert_teacher_blend"]
    evaluator = Evaluator(cache)
    # A teacher's path can start off the current rear axle. Use the actual
    # tracking rollout as another independently rescored, dynamically joined seed.
    _, tracked_reference = evaluator.score(reference, details=True)
    rollout = to_local(tracked_reference[5::5, :3], origin)
    names += ["tracked_pdm_teacher"]
    path_list = [gt.astype(np.float32), timed, reference, blend, rollout]
    tracked_local = to_local(tracked_reference[:, :3], origin)
    output_times = np.arange(1, 9) * 0.5
    for fraction in (0.95, 0.90, 0.85):
        # t'(0)=1: preserve the initial velocity while slowing later progress.
        desired = output_times - (1 - fraction) * output_times**2 / 4
        names.append("teacher_time_%.2f" % fraction)
        path_list.append(resample_recorded(tracked_local, np.arange(41) * 0.1, desired))
    paths = np.stack(path_list)
    policy = TargetPolicy(**policy_values)
    dynamic = cache.ego_state.dynamic_car_state
    velocity = [dynamic.rear_axle_velocity_2d.x, dynamic.rear_axle_velocity_2d.y]
    acceleration = [dynamic.rear_axle_acceleration_2d.x, dynamic.rear_axle_acceleration_2d.y]
    try:
        space = planned_space_checker(cache)
    except MissingPlannedGeometry:
        space = None
    arms, labels = {}, []
    for name, path in zip(names, paths):
        label, simulated = evaluator.score(path, details=True)
        labels.append(label)
        masks = preliminary_rules(path[None], label[None], gt, policy, velocity, acceleration)
        failed = [k for k, value in masks.items() if not value[0]]
        errors = tracking_errors(path, simulated, origin)
        spatial_ok = bool(space(path)) if space is not None else False
        if not spatial_ok:
            failed.append("planned_space" if space else "missing_geometry")
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
        arms[name] = dict(
            pdms=float(label[-1] * 100),
            labels=label.tolist(),
            failed_rules=failed,
            generator_eligible=not failed,
            tracking_errors=list(errors),
            planned_space_ok=spatial_ok,
        )
    result = dict(
        record=record,
        token=record["token"],
        full_count=len(original),
        full_best_pdms=float(original["labels"][:, -1].max() * 100),
        old_ge95=int((original["labels"][:, -1] >= 0.95).sum()),
        max_frame_interval_error_s=float(abs(np.diff(times) - 0.5).max()),
        arms=arms,
    )
    destination = Path(output) / "scenes"
    destination.mkdir(exist_ok=True, parents=True)
    atomic(
        destination / (record["token"] + ".npz"),
        lambda f: np.savez(f, names=names, trajectories=paths, labels=labels),
    )
    write_json(destination / (record["token"] + ".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    import yaml
    from .store import read_json, write_json

    tokens = args.tokens.read_text().split()
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate tokens")
    records = {
        r["token"]: r
        for r in map(json.loads, (args.source / "scenes.jsonl").read_text().splitlines())
    }
    chosen = [records[t] for t in sorted(tokens)]
    if args.limit:
        chosen = chosen[: args.limit]
    policy = yaml.safe_load(args.policy.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(
        version=3,
        source=str(args.source),
        tokens=[r["token"] for r in chosen],
        policy=policy,
        purpose="Official rescoring of recovery seeds; no source changes or training",
    )
    if (args.output / "config.json").exists() and read_json(args.output / "config.json") != config:
        raise ValueError("Resume configuration changed")
    write_json(args.output / "config.json", config)
    results, errors, jobs = [], [], []
    for record in chosen:
        previous = args.output / "scenes" / (record["token"] + ".json")
        if previous.exists():
            results.append(read_json(previous))
        else:
            jobs.append((record, str(args.source), policy, str(args.output)))
    begin = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(inspect_scene, job): job[0]["token"] for job in jobs}
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception:
                error = dict(token=pending[future], traceback=traceback.format_exc())
                errors.append(error)
                print(json.dumps(error), flush=True)
            if (len(results) + len(errors)) % 32 == 0:
                print(
                    json.dumps(
                        dict(
                            processed=len(results),
                            errors=len(errors),
                            elapsed_s=time.time() - begin,
                        )
                    ),
                    flush=True,
                )
    summary = dict(scenes=len(results), errors=errors, elapsed_s=time.time() - begin, arms={})
    if results:
        for name in results[0]["arms"]:
            arm = [r["arms"][name] for r in results]
            reasons = sorted({k for a in arm for k in a["failed_rules"]})
            summary["arms"][name] = dict(
                at_least95=sum(a["pdms"] >= 95 for a in arm),
                generator_eligible=sum(a["generator_eligible"] for a in arm),
                mean_pdms=float(np.mean([a["pdms"] for a in arm])),
                failed_rules={k: sum(k in a["failed_rules"] for a in arm) for k in reasons},
            )
        summary.update(
            full3072_present=sum(r["full_count"] == 3072 for r in results),
            any_seed_ge95=sum(any(a["pdms"] >= 95 for a in r["arms"].values()) for r in results),
            any_generator_eligible_seed=sum(
                any(a["generator_eligible"] for a in r["arms"].values()) for r in results
            ),
        )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("Recovery seed inspection has errors")


if __name__ == "__main__":
    main()
