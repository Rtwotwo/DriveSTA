"""Search timing and geometry around offline references, with unchanged checks."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

from .recovery_seeds import resample_recorded, to_local


def time_profile(times, fraction, bulge, horizon=4.0):
    """Monotone time change on [0,4], with unit initial derivative.

    fraction controls the terminal time; bulge changes intermediate arrival
    times while preserving both endpoint times and endpoint derivatives.
    """
    t = np.asarray(times, float)
    u = t / 4
    result = t - (1 - fraction) * t * t / 4 + bulge * 16 * u * u * (1 - u) * (1 - u)
    if np.any(np.diff(result) <= 0) or result.min() < -1e-9 or result.max() > horizon + 1e-9:
        raise ValueError("Nonmonotone or out-of-range timing request")
    return np.clip(result, 0, horizon)


def joined_reference(times, reference, velocity, acceleration, join_time):
    """Join the measured rear-axle state to the reference with C2 correction."""
    from scipy.interpolate import CubicSpline

    t = np.asarray(times, float)
    curve = CubicSpline(t, np.asarray(reference, float)[:, :2], axis=0)
    d0 = -curve(0.0)
    d1 = (np.asarray(velocity) - curve(0.0, 1)) * join_time
    d2 = (np.asarray(acceleration) - curve(0.0, 2)) * join_time**2 / 2
    coeff = np.stack(
        [d0, d1, d2, -10 * d0 - 6 * d1 - 3 * d2, 15 * d0 + 8 * d1 + 3 * d2, -6 * d0 - 3 * d1 - d2]
    )
    u = np.minimum(t / join_time, 1.0)
    correction = sum(u[:, None] ** k * coeff[k] for k in range(6))
    derivative = sum(k * u[:, None] ** (k - 1) * coeff[k] / join_time for k in range(1, 6))
    derivative[t >= join_time] = 0
    correction[t >= join_time] = 0
    position = curve(t) + correction
    motion = curve(t, 1) + derivative
    heading = np.arctan2(motion[:, 1], motion[:, 0])
    stationary = np.linalg.norm(motion, axis=1) < 0.05
    heading[stationary] = np.asarray(reference)[stationary, 2]
    result = np.column_stack((position, heading))
    result[0] = 0.0
    return result


def blend_paths(a, b, weight):
    result = (1 - weight) * np.asarray(a, float) + weight * np.asarray(b, float)
    delta = np.arctan2(np.sin(b[:, 2] - a[:, 2]), np.cos(b[:, 2] - a[:, 2]))
    result[:, 2] = a[:, 2] + weight * delta
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result.astype(np.float32)


def inspect_one(job):
    record, config = job
    import torch
    from .source import load_scene, Evaluator
    from .filter_targets import (
        TargetPolicy,
        preliminary_rules,
        planned_space_checker,
        tracking_errors,
    )
    from .store import read_json, write_json, atomic

    torch.set_num_threads(1)
    token = record["token"]
    output = Path(config["output"]) / "scenes"
    output.mkdir(parents=True, exist_ok=True)
    if (output / (token + ".json")).exists():
        return read_json(output / (token + ".json"))
    folder = Path(config["source"]) / "scenes" / record["split"] / token
    contract = read_json(folder / "contract.json")
    scene, expert, cache, identity = load_scene(record, [token], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source identity changed")
    with np.load(Path(config["seeds"]) / "scenes" / (token + ".npz"), allow_pickle=False) as z:
        prior = dict(zip(z["names"], z["trajectories"]))
    evaluator = Evaluator(cache)
    pose = cache.ego_state.rear_axle
    origin = [pose.x, pose.y, pose.heading]
    _, sim = evaluator.score(prior["offline_pdm_teacher"], details=True)
    teacher = to_local(sim[:, :3], origin)
    _, sim2 = evaluator.score(prior["tracked_pdm_teacher"], details=True)
    tracked_twice = to_local(sim2[:, :3], origin)
    start = scene.scene_metadata.num_history_frames - 1
    times = (
        np.array([f.timestamp for f in scene.frames[start:]], float) - scene.frames[start].timestamp
    ) / 1e6
    future = scene.get_future_trajectory(num_trajectory_frames=len(times) - 1).poses
    dense_times = np.arange(41) * 0.1
    timed = resample_recorded(np.vstack((np.zeros((1, 3)), future)), times, dense_times)
    parents = [("teacher", teacher), ("tracked_twice", tracked_twice), ("expert", timed)]
    parents += [
        ("blend_%02d" % round(w * 100), blend_paths(teacher, timed, w)) for w in (0.25, 0.5, 0.75)
    ]
    policy = TargetPolicy(**config["policy"])
    space = planned_space_checker(cache)
    dyn = cache.ego_state.dynamic_car_state
    velocity = [dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y]
    acceleration = [dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y]
    reference_queries = 2
    if config.get("family", "motion") == "joined":
        from nuplan.common.actor_state.state_representation import TimePoint

        horizon = min(
            5.0, (cache.trajectory.end_time.time_us - cache.ego_state.time_point.time_us) / 1e6
        )
        dense_times = np.arange(round(horizon * 10) + 1) * 0.1
        requests = [
            TimePoint(cache.ego_state.time_point.time_us + round(t * 1e6)) for t in dense_times
        ]
        states = cache.trajectory.get_state_at_times(requests)
        reference = to_local(
            np.array([[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading] for s in states]), origin
        )
        parents = [
            (
                "join_%03d" % round(j * 100),
                joined_reference(dense_times, reference, velocity, acceleration, j),
            )
            for j in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
        ]
        # A constant offset produces a parallel path; independently validate it
        # just like the finite-time joins, including full-body road coverage.
        shifted = reference.copy()
        shifted[:, :2] -= shifted[0, :2]
        shifted[0] = 0
        parents.append(("shifted_reference", shifted))
    if config.get("family") == "projected":
        if not config.get("candidate_root"):
            raise ValueError("Projected search requires a scored candidate root")
        with np.load(
            Path(config["candidate_root"]) / "scenes" / (token + ".npz"), allow_pickle=False
        ) as z:
            ordered = np.argsort(-z["labels"][:, -1], kind="stable")
            selected = []
            for i in ordered:
                path = z["trajectories"][i]
                if (
                    not selected
                    or min(np.linalg.norm(path[:, :2] - p[:, :2], axis=1).mean() for p in selected)
                    >= 0.25
                ):
                    selected.append(path.copy())
                if len(selected) == 4:
                    break
        parents = []
        for i, path in enumerate(selected):
            for iteration in range(1, 4):
                _, tracked = evaluator.score(path, details=True)
                reference_queries += 1
                local = to_local(tracked[:, :3], origin)
                parents.append(("project_%d_%d" % (i, iteration), local))
                path = local[5::5]
    # All-one labels only bypass score-dependent prechecks; every recorded
    # candidate receives fresh official labels before acceptance.
    placeholder = np.ones((1, 7))
    names, paths, labels, arms = [], [], [], {}
    failures, prechecks, seen, eligible = {}, 0, set(), 0
    fractions = [1.0] + [1 - i * 0.005 for i in range(1, 25)] + [0.86, 0.84]
    if config.get("family", "motion") == "joined":
        fractions += [1 + i * 0.01 for i in range(1, 11)]
    params = sorted(
        ((f, b) for b in (0.0, -0.1, -0.2, -0.35, -0.5, 0.1, 0.2) for f in fractions),
        key=lambda p: (10 * abs(1 - p[0]) + abs(p[1]), p),
    )
    for fraction, bulge in params:
        try:
            desired = time_profile(np.arange(41) * 0.1, fraction, bulge, dense_times[-1])
        except ValueError:
            continue
        for parent_name, parent in parents:
            path = resample_recorded(parent, dense_times, desired[5::5])
            key = path.tobytes()
            if key in seen:
                continue
            seen.add(key)
            prechecks += 1
            masks = preliminary_rules(
                path[None], placeholder, expert, policy, velocity, acceleration
            )
            failed = [k for k, mask in masks.items() if not mask[0]]
            if not failed and not space(path):
                failed = ["planned_space"]
            if failed:
                for reason in failed:
                    failures[reason] = failures.get(reason, 0) + 1
                continue
            label, tracked = evaluator.score(path, details=True)
            masks = preliminary_rules(
                path[None], label[None], expert, policy, velocity, acceleration
            )
            failed = [k for k, mask in masks.items() if not mask[0]]
            errors = tracking_errors(path, tracked, origin)
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
            name = "%s_f%04d_b%+04d" % (parent_name, round(fraction * 1000), round(bulge * 100))
            arms[name] = dict(
                pdms=float(label[-1] * 100),
                labels=label.tolist(),
                failed_rules=failed,
                generator_eligible=not failed,
                tracking_errors=list(errors),
                planned_space_ok=True,
            )
            names.append(name)
            paths.append(path)
            labels.append(label)
            eligible += not failed
            if eligible >= 3 or len(paths) >= config["max_queries"]:
                break
        if eligible >= 3 or len(paths) >= config["max_queries"]:
            break
    atomic(
        output / (token + ".npz"),
        lambda f: np.savez(
            f,
            names=np.asarray(names, dtype="U80"),
            trajectories=np.asarray(paths, np.float32).reshape(-1, 8, 3),
            labels=np.asarray(labels, float).reshape(-1, 7),
        ),
    )
    result = dict(
        token=token,
        record=record,
        arms=arms,
        queries=len(paths),
        prechecks=prechecks,
        eligible=eligible,
        precheck_failures=failures,
        reference_rollout_queries=reference_queries,
        search_exhausted=eligible < 3 and len(paths) < config["max_queries"],
    )
    write_json(output / (token + ".json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "seeds", "tokens", "policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-queries", type=int, default=160)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--family", choices=("motion", "joined", "projected"), default="motion")
    p.add_argument("--candidate-root", type=Path)
    args = p.parse_args()
    import yaml
    from .store import read_json, write_json

    tokens = sorted(args.tokens.read_text().split())
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate input token")
    if args.limit:
        tokens = tokens[: args.limit]
    records = {
        r["token"]: r
        for r in map(json.loads, (args.source / "scenes.jsonl").read_text().splitlines())
    }
    config = dict(
        version=4,
        family=args.family,
        candidate_root=str(args.candidate_root) if args.candidate_root else None,
        source=str(args.source),
        seeds=str(args.seeds),
        output=str(args.output),
        tokens=tokens,
        max_queries=args.max_queries,
        policy=yaml.safe_load(args.policy.read_text()),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists() and read_json(args.output / "config.json") != config:
        raise ValueError("Resume configuration differs")
    write_json(args.output / "config.json", config)
    results, errors, start = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        jobs = {pool.submit(inspect_one, (records[t], config)): t for t in tokens}
        for future in as_completed(jobs):
            try:
                r = future.result()
                results.append(r)
                print(
                    json.dumps({k: r[k] for k in ("token", "queries", "prechecks", "eligible")}),
                    flush=True,
                )
            except Exception:
                e = dict(token=jobs[future], traceback=traceback.format_exc())
                errors.append(e)
                print(json.dumps(e), flush=True)
    passed = [r["token"] for r in results if r["eligible"]]
    summary = dict(
        scenes=len(results),
        eligible_scenes=len(passed),
        eligible_tokens=passed,
        queries=sum(r["queries"] for r in results),
        errors=errors,
        elapsed_s=time.time() - start,
    )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("Motion search contains errors")


if __name__ == "__main__":
    main()
