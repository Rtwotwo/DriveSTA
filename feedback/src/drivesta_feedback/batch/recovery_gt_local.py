"""Bounded perturbations of recorded experts, retaining every official query."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

from .filter_targets import wrap
from .recovery_seeds import resample_recorded


def perturb_expert(poses, times, coefficients):
    """Monotone timing and a small smooth displacement in the expert frame.

    Coefficients: time shift, terminal along/across, middle along/across,
    middle heading, optionally terminal time shift. Initial displacement and
    its derivative vanish. Every requested time must be present in the log.
    """
    c = np.asarray(coefficients, float)
    if c.shape not in ((6,), (7,)) or not np.isfinite(c).all():
        raise ValueError("Six or seven finite coefficients required")
    advance = c[6] if len(c) == 7 else 0.0
    t = np.arange(1, 9) * 0.5
    u = t / 4
    # Optional terminal advance uses later recorded GT, never extrapolation.
    w = 16 * u * u * (1 - u) ** 2
    desired = t + c[0] * w + advance * u * u
    dense_t = np.linspace(0, 4, 161)
    dense_u = dense_t / 4
    dense_w = 16 * dense_u**2 * (1 - dense_u) ** 2
    if np.any(np.diff(dense_t + c[0] * dense_w + advance * dense_u**2) <= 0):
        raise ValueError("Nonmonotone timing")
    baseline = resample_recorded(poses, times, desired).astype(float)
    beta = u**3 * (10 - 15 * u + 6 * u * u)
    along, across = beta * c[1] + w * c[3], beta * c[2] + w * c[4]
    tangent = np.column_stack((np.cos(baseline[:, 2]), np.sin(baseline[:, 2])))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    baseline[:, :2] += along[:, None] * tangent + across[:, None] * normal
    baseline[:, 2] = wrap(baseline[:, 2] + w * c[5])
    return baseline.astype(np.float32)


def recover(job):
    record, config = job
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    import torch
    from scipy.optimize import minimize
    from scipy.stats import qmc
    from .source import load_scene, Evaluator
    from .filter_targets import (
        TargetPolicy,
        preliminary_rules,
        planned_space_checker,
        tracking_errors,
    )
    from .recovery_projection_search import motion_slack
    from .recovery_rollout_search import Geometry, comfort_slack, direction_slack, SearchStop
    from .store import read_json, write_json, atomic
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        BBCoordsIndex,
        EgoAreaIndex,
    )

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
    start = scene.scene_metadata.num_history_frames - 1
    times = (
        np.array([f.timestamp for f in scene.frames[start:]], float) - scene.frames[start].timestamp
    ) / 1e6
    future = scene.get_future_trajectory(num_trajectory_frames=len(times) - 1).poses
    poses = np.vstack((np.zeros((1, 3)), future))
    policy = TargetPolicy(**config["policy"])
    evaluator, geometry, space = Evaluator(cache), Geometry(cache), planned_space_checker(cache)
    origin = np.array(
        [
            cache.ego_state.rear_axle.x,
            cache.ego_state.rear_axle.y,
            cache.ego_state.rear_axle.heading,
        ]
    )
    dyn = cache.ego_state.dynamic_car_state
    velocity = np.array([dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y])
    acceleration = np.array([dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y])
    initial_ok = bool(space(np.zeros((8, 3), np.float32)))
    limits = np.array(
        [
            policy.max_tracking_ade_m,
            policy.max_tracking_error_m,
            policy.max_tracking_heading_error_rad,
        ]
    )
    bounds = np.array(config["coefficient_bounds"], float)
    dimensions = len(bounds)
    if dimensions == 7:
        bounds[6] = min(bounds[6], max(0.0, times[-1] - 4.0 - 1e-5))
    timed_gt = resample_recorded(poses, times)
    paths, labels, names, parameters, arms, memo, runs = [], [], [], [], {}, {}, []
    eligible = 0

    def query(c, explicit=None, name=None):
        nonlocal eligible
        path = (
            perturb_expert(poses, times, c)
            if explicit is None
            else np.asarray(explicit, np.float32)
        )
        key = path.tobytes()
        if key in memo:
            return memo[key]
        if len(paths) >= config["max_queries"] or eligible >= 3:
            raise SearchStop()
        label, tracked = evaluator.score(path, details=True)
        errors = np.array(tracking_errors(path, tracked, origin))
        masks = preliminary_rules(path[None], label[None], expert, policy, velocity, acceleration)
        failed = [k for k, m in masks.items() if not m[0]]
        spatial_ok = bool(space(path))
        if not spatial_ok:
            failed.append("planned_space")
        if np.any(errors > limits):
            failed.append("tracking")
        name = name or "gtlocal_%05d" % len(paths)
        planned = geometry.clearance(geometry.states(path))
        actual = geometry.clearance(tracked)
        scorer = evaluator.scorer
        ddc = direction_slack(
            scorer._ego_coords[1, :, BBCoordsIndex.CENTER],
            scorer._ego_areas[1, :, EgoAreaIndex.ONCOMING_TRAFFIC],
            horizon=int(scorer._config.driving_direction_horizon / 0.1),
            threshold_m=scorer._config.driving_direction_compliance_threshold,
        )
        slack = np.r_[
            motion_slack(path, velocity, acceleration, policy),
            planned,
            actual[:41],
            ddc,
            1 - errors / (limits * 0.98),
            comfort_slack(tracked),
        ]
        deviation = np.linalg.norm(path[:, :2] - timed_gt[:, :2], axis=1).max()
        if config.get("max_gt_deviation_m") is not None:
            slack = np.r_[slack, 1 - deviation / config["max_gt_deviation_m"]]
            if deviation > config["max_gt_deviation_m"]:
                failed.append("gt_neighborhood_bound")
        arms[name] = dict(
            pdms=float(label[-1] * 100),
            labels=label.tolist(),
            failed_rules=failed,
            generator_eligible=not failed,
            tracking_errors=errors.tolist(),
            planned_space_ok=spatial_ok,
            search_minimum_slack=float(slack.min()),
            max_gt_deviation_m=float(deviation),
        )
        paths.append(path)
        labels.append(label)
        names.append(name)
        parameters.append(np.asarray(c, float))
        eligible += not failed
        value = (-float(label[2]), slack)
        memo[key] = value
        return value

    query(np.zeros(dimensions), explicit=expert, name="raw_expert")
    query(np.zeros(dimensions), name="timestamp_expert")
    designs = (
        qmc.Sobol(dimensions, scramble=True, seed=config["seed"]).random_base2(
            config["sobol_power"]
        )
        * 2
        - 1
    )
    designs *= bounds
    try:
        # Structured signed perturbations retain interpretable small changes.
        for scale in (0.25, 0.5, 1.0):
            for dim in range(dimensions):
                for sign in (-1, 1):
                    c = np.zeros(dimensions)
                    c[dim] = bounds[dim] * scale * sign
                    query(c)
        for c in designs:
            query(c)
        if initial_ok and config["iterations"]:
            ranked = sorted(
                range(1, len(names)),
                key=lambda i: (
                    len(set(arms[names[i]]["failed_rules"]) - {"pdm"}),
                    -arms[names[i]]["search_minimum_slack"],
                    -labels[i][-1],
                ),
            )
            starts = [np.zeros(dimensions)] + [parameters[i] for i in ranked[:3]]
            for c in starts:
                result = minimize(
                    lambda x: query(x)[0],
                    c,
                    method="SLSQP",
                    bounds=list(zip(-bounds, bounds)),
                    constraints=[dict(type="ineq", fun=lambda x: query(x)[1])],
                    options=dict(maxiter=config["iterations"], ftol=1e-7, eps=0.002),
                )
                query(result.x)
                runs.append(
                    dict(
                        success=bool(result.success),
                        message=str(result.message),
                        iterations=int(result.nit),
                    )
                )
    except SearchStop:
        runs.append(dict(stopped="query budget or three qualified paths"))
    atomic(
        out / (token + ".npz"),
        lambda f: np.savez(
            f,
            names=np.asarray(names, dtype="U80"),
            trajectories=np.asarray(paths, np.float32),
            labels=np.asarray(labels, float),
            coefficients=np.asarray(parameters, float),
            recorded_poses=poses,
            recorded_times=times,
        ),
    )
    row = dict(
        token=token,
        record=record,
        queries=len(paths),
        eligible=eligible,
        arms=arms,
        initial_spatial_ok=initial_ok,
        optimizations=runs,
        source_identity_verified=True,
        official_policy_unchanged=True,
        all_scored_candidates_retained=True,
    )
    write_json(out / (token + ".json"), row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokens", "policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-queries", type=int, default=4000)
    p.add_argument("--sobol-power", type=int, default=10)
    p.add_argument("--iterations", type=int, default=70)
    p.add_argument("--terminal-time-shift", type=float, default=0.0)
    a = p.parse_args()
    import yaml
    from .store import read_json, write_json

    tokens = sorted(a.tokens.read_text().split())
    if (
        not tokens
        or len(tokens) != len(set(tokens))
        or not 0 <= a.sobol_power <= 14
        or min(a.workers, a.max_queries) < 1
        or a.iterations < 0
        or not 0 <= a.terminal_time_shift <= 0.75
    ):
        raise ValueError("Invalid token list or search budget")
    config = dict(
        version=1,
        source=str(a.source),
        output=str(a.output),
        tokens=tokens,
        policy=yaml.safe_load(a.policy.read_text()),
        coefficient_bounds=[0.3, 1.5, 0.5, 0.75, 0.35, 0.1],
        seed=20261019,
        sobol_power=a.sobol_power,
        max_queries=a.max_queries,
        iterations=a.iterations,
    )
    if a.terminal_time_shift:
        config.update(version=2, max_gt_deviation_m=6.0)
        config["coefficient_bounds"].append(a.terminal_time_shift)
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "config.json").exists() and read_json(a.output / "config.json") != config:
        raise ValueError("Resume configuration differs")
    write_json(a.output / "config.json", config)
    records = {
        r["token"]: r for r in map(json.loads, (a.source / "scenes.jsonl").read_text().splitlines())
    }
    start = time.time()
    rows = []
    errors = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(recover, (records[t], config)): t for t in tokens}
        for f in as_completed(futures):
            try:
                row = f.result()
                rows.append(row)
                print(
                    json.dumps(
                        {k: row[k] for k in ("token", "queries", "eligible", "initial_spatial_ok")}
                    ),
                    flush=True,
                )
            except Exception:
                error = dict(token=futures[f], traceback=traceback.format_exc())
                errors.append(error)
                print(json.dumps(error), flush=True)
    summary = dict(
        scenes=len(rows),
        queries=sum(r["queries"] for r in rows),
        eligible_tokens=[r["token"] for r in rows if r["eligible"]],
        errors=errors,
        elapsed_s=time.time() - start,
    )
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("GT local search has errors")


if __name__ == "__main__":
    main()
