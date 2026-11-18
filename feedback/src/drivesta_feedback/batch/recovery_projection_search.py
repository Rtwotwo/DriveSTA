"""Recover difficult references by constrained waypoint repair and replay.

Optimization proposes trajectories only. Official labels and the unchanged
driving policy decide acceptance; no optimizer objective is a training label.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

from .filter_targets import wrap


def motion_slack(path, velocity, acceleration, policy, margin=0.98):
    """Continuous, conservative counterparts of the existing motion checks."""
    full = np.vstack((np.zeros((1, 3)), np.asarray(path).reshape(8, 3)))
    v = np.diff(full[:, :2], axis=0) / 0.5
    speed = np.linalg.norm(v, axis=1)
    a = np.diff(np.vstack((velocity, v)), axis=0) / 0.5
    j = np.diff(np.vstack((acceleration, a)), axis=0) / 0.5
    dh = wrap(np.diff(full[:, 2]))
    mid = full[:-1, 2] + dh / 2
    signed = np.sum(v * np.column_stack((np.cos(mid), np.sin(mid))), axis=1)
    heading = abs(wrap(np.arctan2(v[:, 1], v[:, 0]) - mid))
    moving = speed > policy.moving_speed_mps
    return np.concatenate(
        (
            1 - speed / (policy.max_speed_mps * margin),
            1 - np.linalg.norm(a, axis=1) / (policy.max_accel_mps2 * margin),
            1 - np.linalg.norm(j, axis=1) / (policy.max_jerk_mps3 * margin),
            1 - np.where(moving, abs(dh) / 0.5, 0) / (policy.max_yaw_rate_rps * margin),
            1 - speed * abs(dh) / 0.5 / (policy.max_lateral_accel_mps2 * margin),
            1 - np.where(moving, heading, 0) / (policy.max_heading_motion_error_rad * margin),
            (signed - policy.min_signed_speed_mps * margin) / max(policy.max_speed_mps, 1),
            [
                1 - np.maximum(-signed, 0).sum() * 0.5 / (policy.max_backward_distance_m * margin),
                1
                - np.linalg.norm(v[0] - velocity)
                / (policy.max_initial_velocity_error_mps * margin),
            ],
        )
    )


def project_motion(target, velocity, acceleration, policy, iterations=100):
    """Find a nearby plan within existing discrete motion limits."""
    from scipy.optimize import minimize

    target = np.asarray(target, float).copy()
    target[:, 2] = np.unwrap(np.r_[0.0, target[:, 2]])[1:]

    def objective(flat):
        delta = flat.reshape(8, 3) - target
        return np.mean(delta[:, :2] ** 2) + np.mean(delta[:, 2] ** 2) * 4

    def gradient(flat):
        delta = flat.reshape(8, 3) - target
        delta[:, :2] /= 8
        delta[:, 2] *= 1.0
        return delta.ravel()

    lower = target - np.array([3.0, 3.0, 0.45])
    upper = target + np.array([3.0, 3.0, 0.45])
    result = minimize(
        objective,
        target.ravel(),
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(lower.ravel(), upper.ravel())),
        constraints=[
            dict(type="ineq", fun=lambda x: motion_slack(x, velocity, acceleration, policy))
        ],
        options=dict(maxiter=iterations, ftol=1e-7),
    )
    path = result.x.reshape(8, 3).copy()
    path[:, 2] = wrap(path[:, 2])
    return path.astype(np.float32), dict(
        success=bool(result.success),
        iterations=int(result.nit),
        objective=float(result.fun),
        minimum_slack=float(motion_slack(path, velocity, acceleration, policy).min()),
    )


def displaced_target(path, longitudinal, lateral):
    """Smooth endpoint displacement in the path's local tangent frame."""
    p = np.asarray(path, float).copy()
    u = np.arange(1, 9) / 8
    beta = u**3 * (10 - 15 * u + 6 * u * u)
    tangent = np.column_stack((np.cos(p[:, 2]), np.sin(p[:, 2])))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    p[:, :2] += beta[:, None] * (longitudinal * tangent + lateral * normal)
    return p.astype(np.float32)


def local_displacement(path, center_s, amplitude, lateral=False):
    """Change a short part of the path, with zero change at the current pose."""
    p = np.asarray(path, float).copy()
    times = np.arange(1, 9) * 0.5
    u = np.clip(abs(times - center_s) / 1.5, 0.0, 1.0)
    weight = 0.5 * (1 + np.cos(np.pi * u))
    # Initial displacement and derivative vanish at t=0.
    weight *= np.minimum(times, 1.0) ** 2
    angle = p[:, 2] + (np.pi / 2 if lateral else 0.0)
    p[:, :2] += (amplitude * weight)[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    return p.astype(np.float32)


def load_parents(token, roots):
    choices = []
    for root in roots:
        path = Path(root) / "scenes" / (token + ".npz")
        if not path.exists():
            continue
        row = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as z:
            for name, trajectory, label in zip(z["names"], z["trajectories"], z["labels"]):
                arm = row["arms"][str(name)]
                errors = arm.get("tracking_errors", [10, 10, 10])
                cost = -label[-1] * 100 + 2 * max(errors[0] - 1, 0) + max(errors[1] - 2, 0)
                cost += 8 * ("planned_space" in arm["failed_rules"])
                cost += 3 * len(set(arm["failed_rules"]) - {"pdm", "tracking", "planned_space"})
                choices.append((cost, str(Path(root).name) + "/" + str(name), trajectory.copy()))
    chosen = []
    for cost, name, path in sorted(choices, key=lambda x: x[0]):
        if all(np.linalg.norm(path[:, :2] - p[:, :2], axis=1).mean() >= 0.25 for _, p in chosen):
            chosen.append((name, path))
        if len(chosen) == 10:
            break
    return chosen


def recover(job):
    record, config = job
    import torch
    from .source import load_scene, Evaluator
    from .filter_targets import (
        TargetPolicy,
        preliminary_rules,
        planned_space_checker,
        tracking_errors,
    )
    from .recovery_seeds import to_local
    from .recovery_motion_search import blend_paths
    from .store import read_json, write_json, atomic

    torch.set_num_threads(1)
    token = record["token"]
    out = Path(config["output"]) / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    if (out / (token + ".json")).exists():
        return read_json(out / (token + ".json"))
    folder = Path(config["source"]) / "scenes" / record["split"] / token
    contract = read_json(folder / "contract.json")
    scene, expert, cache, identity = load_scene(record, [token], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source identity changed")
    policy = TargetPolicy(**config["policy"])
    evaluator, space = Evaluator(cache), planned_space_checker(cache)
    pose, dyn = cache.ego_state.rear_axle, cache.ego_state.dynamic_car_state
    origin = [pose.x, pose.y, pose.heading]
    velocity = np.array([dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y])
    acceleration = np.array([dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y])
    parents = load_parents(token, config["candidate_roots"])
    paths, labels, names, arms, optimization = [], [], [], {}, []
    seen, pre_failures, beam = set(), {}, []
    eligible = 0
    strong_eligible = 0

    def evaluate(target, name, repair=True):
        nonlocal eligible, strong_eligible
        if len(paths) >= config["max_queries"]:
            return
        if repair:
            path, result = project_motion(
                target, velocity, acceleration, policy, config["iterations"]
            )
            optimization.append(dict(name=name, **result))
        else:
            path = np.asarray(target, np.float32)
        key = path.tobytes()
        if key in seen:
            return
        seen.add(key)
        masks = preliminary_rules(
            path[None], np.ones((1, 7)), expert, policy, velocity, acceleration
        )
        failed = [k for k, m in masks.items() if not m[0]]
        if not space(path):
            failed.append("planned_space")
        if failed:
            for k in failed:
                pre_failures[k] = pre_failures.get(k, 0) + 1
            return
        label, tracked = evaluator.score(path, details=True)
        masks = preliminary_rules(path[None], label[None], expert, policy, velocity, acceleration)
        failed = [k for k, m in masks.items() if not m[0]]
        errors = tracking_errors(path, tracked, origin)
        limits = [
            policy.max_tracking_ade_m,
            policy.max_tracking_error_m,
            policy.max_tracking_heading_error_rad,
        ]
        if any(a > b for a, b in zip(errors, limits)):
            failed.append("tracking")
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
        strong_eligible += not failed and label[-1] >= config.get("seed_score_goal", 0.95)
        merit = float(label[-1] * 100 - 5 * max(errors[0] - 1, 0) - 2 * max(errors[1] - 2, 0))
        merit -= 10 * len(set(failed) - {"pdm", "tracking"})
        beam.append((merit, name, path, to_local(tracked[5::5, :3], origin)))

    offsets = [(0, 0), (1, 0), (-1, 0), (2, 0), (0, 0.5), (0, -0.5), (4, 0)]
    if config.get("coordinate_search", False):
        offsets = [(0, 0)]
    if config.get("wide", False):
        offsets += [(a, d) for a in (2.0, 4.0, 6.0, 8.0) for d in (-0.75, 0.75)]
        offsets += [(0.0, -1.5), (0.0, 1.5), (6.0, 0), (8.0, 0)]
    for advance, lateral in offsets:
        for i, (parent_name, parent) in enumerate(parents):
            evaluate(
                displaced_target(parent, advance, lateral),
                "parent%02d_a%g_l%g" % (i, advance, lateral),
            )
            if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                break
        if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
            break
    # Partially move the plan toward its actual rollout to reduce tracking
    # mismatch, retaining progress through a separate endpoint displacement.
    if strong_eligible < 3 and not config.get("coordinate_search", False):
        best = sorted(beam, key=lambda x: -x[0])[:8]
        for w in (0.25, 0.5, 0.75, 1.0):
            for advance in (0.0, 1.0, 2.0):
                for i, (_, parent_name, parent, tracked) in enumerate(best):
                    target = displaced_target(blend_paths(parent, tracked, w), advance, 0.0)
                    evaluate(target, "feedback%02d_w%g_a%g" % (i, w, advance))
                    if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                        break
                if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                    break
            if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                break
    if strong_eligible < 3 and config.get("coordinate_search", False):
        for round_id, scale in enumerate((0.5, 0.25, 0.1, 0.05)):
            for center in (4.0, 3.0, 2.0, 1.0):
                for lateral in (False, True):
                    # Refresh the incumbent after each coordinate. The score
                    # used here is only an offline search objective.
                    if not beam:
                        break
                    ordered = sorted(beam, key=lambda x: -x[0])
                    bases = [ordered[0]]
                    for b in ordered[1:]:
                        if np.linalg.norm(b[2][:, :2] - bases[0][2][:, :2], axis=1).mean() >= 0.25:
                            bases.append(b)
                            break
                    for i, (_, parent_name, parent, tracked) in enumerate(bases):
                        for sign in (1, -1):
                            target = local_displacement(parent, center, scale * sign, lateral)
                            name = "walk%d_c%d_l%d_i%d_s%d" % (round_id, center, lateral, i, sign)
                            evaluate(target, name)
                            if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                                break
                        if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                            break
                    if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                        break
                if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                    break
            if strong_eligible >= 3 or len(paths) >= config["max_queries"]:
                break
    atomic(
        out / (token + ".npz"),
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
        eligible=eligible,
        precheck_failures=pre_failures,
        optimizations=optimization,
        parent_sources=[n for n, p in parents],
        no_candidate_score_inherited=True,
    )
    write_json(out / (token + ".json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokens", "policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate-roots", nargs="+", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-queries", type=int, default=160)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--wide", action="store_true")
    p.add_argument("--coordinate-search", action="store_true")
    p.add_argument(
        "--seed-score-goal",
        type=float,
        default=0.95,
        help="Optional higher search stopping target; acceptance policy is unchanged",
    )
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    import yaml
    from .store import read_json, write_json

    tokens = sorted(a.tokens.read_text().split())
    if len(tokens) != len(set(tokens)):
        raise ValueError("Repeated token")
    if a.limit:
        tokens = tokens[: a.limit]
    records = {
        r["token"]: r for r in map(json.loads, (a.source / "scenes.jsonl").read_text().splitlines())
    }
    if a.wide and a.coordinate_search:
        raise ValueError("Choose one search family per run")
    if not 0.95 <= a.seed_score_goal <= 1.0:
        raise ValueError("Seed score goal must be in [.95, 1]")
    config = dict(
        version=4,
        seed_score_goal=a.seed_score_goal,
        wide=a.wide,
        coordinate_search=a.coordinate_search,
        source=str(a.source),
        output=str(a.output),
        tokens=tokens,
        candidate_roots=[str(x) for x in a.candidate_roots],
        max_queries=a.max_queries,
        iterations=a.iterations,
        policy=yaml.safe_load(a.policy.read_text()),
    )
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "config.json").exists() and read_json(a.output / "config.json") != config:
        raise ValueError("Resume configuration differs")
    write_json(a.output / "config.json", config)
    results, errors, start = [], [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        jobs = {pool.submit(recover, (records[t], config)): t for t in tokens}
        for f in as_completed(jobs):
            try:
                row = f.result()
                results.append(row)
                print(json.dumps({k: row[k] for k in ("token", "queries", "eligible")}), flush=True)
            except Exception:
                error = dict(token=jobs[f], traceback=traceback.format_exc())
                errors.append(error)
                print(json.dumps(error), flush=True)
    passed = [r["token"] for r in results if r["eligible"]]
    summary = dict(
        scenes=len(results),
        eligible_scenes=len(passed),
        eligible_tokens=passed,
        queries=sum(r["queries"] for r in results),
        errors=errors,
        elapsed_s=time.time() - start,
    )
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("Projection search contains errors")


if __name__ == "__main__":
    main()
