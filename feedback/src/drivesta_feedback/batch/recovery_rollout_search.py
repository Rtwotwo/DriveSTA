"""Offline constrained search using actual official tracking feedback.

All optimizer probes retain their official labels, including rejected paths.
The existing target policy, rather than optimizer success, decides acceptance.
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
from .recovery_projection_search import motion_slack, load_parents


def signed_clearance(polygon, xy):
    """Positive inside, negative outside; measured against the same polygon."""
    import shapely

    points = shapely.points(np.asarray(xy).reshape(-1, 2))
    distance = shapely.distance(polygon.boundary, points)
    return np.where(shapely.covers(polygon, points), distance, -distance)


def comfort_slack(states, margin=0.98):
    """Use official extractors and thresholds; retain a search-only margin."""
    from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_comfort_metrics as m

    states = np.asarray(states)[None]
    n = states.shape[1]
    times = np.arange(n, dtype=float) * 0.1
    lon = m._extract_ego_acceleration(states, "x", window_length=n)[0]
    lat = m._extract_ego_acceleration(states, "y", window_length=n)[0]
    jerk = m._extract_ego_jerk(states, "magnitude", times, window_length=n)[0]
    lon_jerk = m._extract_ego_jerk(states, "x", times, window_length=n)[0]
    yaw_acc = m._extract_ego_yaw_rate(states, times, deriv_order=2, poly_order=3, window_length=n)[
        0
    ]
    yaw_rate = m._extract_ego_yaw_rate(states, times, window_length=n)[0]
    return np.r_[
        (lon - m.min_lon_accel * margin) / abs(m.min_lon_accel * margin),
        1 - lon / (m.max_lon_accel * margin),
        1 - abs(lat) / (m.max_abs_lat_accel * margin),
        1 - abs(jerk) / (m.max_abs_mag_jerk * margin),
        1 - abs(lon_jerk) / (m.max_abs_lon_jerk * margin),
        1 - abs(yaw_acc) / (m.max_abs_yaw_accel * margin),
        1 - abs(yaw_rate) / (m.max_abs_yaw_rate * margin),
    ]


def direction_slack(center_xy, outside_route, horizon=10, threshold_m=2.0, margin=0.99):
    """Continuous distance term using the official time mask and window."""
    progress = np.r_[0.0, np.linalg.norm(np.diff(center_xy, axis=0), axis=1)]
    progress[~np.asarray(outside_route, bool)] = 0.0
    windows = np.array([progress[max(0, i - horizon) : i + 1].sum() for i in range(len(progress))])
    return 1 - windows / (threshold_m * margin)


class Geometry:
    def __init__(self, cache):
        import shapely
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        dm = cache.drivable_area_map
        ids = dm.get_indices_of_map_type(
            [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.CARPARK_AREA,
            ]
        )
        self.road = shapely.union_all([dm[dm.tokens[i]] for i in ids])
        self.route = shapely.union_all([dm[t] for t in dm.tokens if t in cache.route_lane_ids])
        if self.road.is_empty or self.route.is_empty:
            raise ValueError("Missing source map geometry")
        self.vehicle = get_pacifica_parameters()
        self.sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.cache = cache

    def states(self, path):
        from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
        from navsim.common.dataclasses import Trajectory

        plan = transform_trajectory(Trajectory(path), self.cache.ego_state)
        return get_trajectory_as_array(plan, self.sampling, self.cache.ego_state.time_point)

    def clearance(self, states):
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
            state_array_to_coords_array,
        )
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex

        coords = state_array_to_coords_array(states[None], self.vehicle)[0]
        road = signed_clearance(self.road, coords[:, :4]).reshape(-1, 4).min(axis=1)
        route = signed_clearance(self.route, coords[:, BBCoordsIndex.CENTER])
        return np.r_[road, route]


class SearchStop(Exception):
    pass


def spline_correction(parent, parameters):
    """Twelve correction variables; continuous correction/slope vanish at t=0.

    The saved eight-waypoint plan is still interpolated and checked by the
    original initial-motion policy; no new initial-velocity claim is made.
    """
    from scipy.interpolate import CubicSpline

    parent = np.asarray(parent, float)
    knots = np.vstack((np.zeros((1, 3)), np.asarray(parameters).reshape(4, 3)))
    knots[:, 2] *= 0.1
    spline = CubicSpline(np.arange(5.0), knots, bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
    return parent + spline(np.arange(1, 9) * 0.5)


def feasibility_parents(token, roots, count):
    """Start from driving-feasible paths before improving their progress."""
    choices = []
    for root in roots:
        p = Path(root) / "scenes" / (token + ".npz")
        if not p.exists():
            continue
        row = json.loads(p.with_suffix(".json").read_text())
        with np.load(p, allow_pickle=False) as z:
            for name, path, label in zip(z["names"], z["trajectories"], z["labels"]):
                arm = row["arms"][str(name)]
                failed = set(arm["failed_rules"]) - {"pdm"}
                error = arm.get("tracking_errors", [10, 10, 10])
                rank = (len(failed), max(error[0] - 1, 0) + max(error[1] - 2, 0), -label[-1])
                choices.append((rank, str(Path(root).name) + "/" + str(name), path.copy()))
    selected = []
    for _, name, path in sorted(choices, key=lambda x: x[0]):
        if all(np.linalg.norm(path[:, :2] - q[:, :2], axis=1).mean() >= 0.25 for _, q in selected):
            selected.append((name, path))
        if len(selected) == count:
            break
    return selected


def recover(job):
    record, config = job
    import torch
    from scipy.optimize import minimize
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
    out = Path(config["output"]) / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    if (out / (token + ".json")).exists():
        return read_json(out / (token + ".json"))
    folder = Path(config["source"]) / "scenes" / record["split"] / token
    contract = read_json(folder / "contract.json")
    _, expert, cache, identity = load_scene(record, [token], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source identity changed")
    policy = TargetPolicy(**config["policy"])
    evaluator, geometry, space = Evaluator(cache), Geometry(cache), planned_space_checker(cache)
    pose, dyn = cache.ego_state.rear_axle, cache.ego_state.dynamic_car_state
    origin = [pose.x, pose.y, pose.heading]
    velocity = np.array([dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y])
    acceleration = np.array([dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y])
    if not space(np.zeros((8, 3), np.float32)):
        raise ValueError("Initial pose fails unchanged spatial policy; audit route first")
    parents = (
        feasibility_parents(token, config["candidate_roots"], config["parents"])
        if config.get("parent_selection") == "feasibility"
        else load_parents(token, config["candidate_roots"])[: config["parents"]]
    )
    paths, labels, names, arms, runs, memo = [], [], [], {}, [], {}
    eligible = 0
    strong = 0

    def query(flat):
        nonlocal eligible, strong
        path = np.asarray(flat, np.float32).reshape(8, 3).copy()
        path[:, 2] = wrap(path[:, 2])
        key = path.tobytes()
        if key in memo:
            return memo[key]
        if len(paths) >= config["max_queries"] or strong >= 3:
            raise SearchStop()
        label, tracked = evaluator.score(path, details=True)
        errors = np.asarray(tracking_errors(path, tracked, origin))
        limits = np.array(
            [
                policy.max_tracking_ade_m,
                policy.max_tracking_error_m,
                policy.max_tracking_heading_error_rad,
            ]
        )
        masks = preliminary_rules(path[None], label[None], expert, policy, velocity, acceleration)
        failed = [k for k, mask in masks.items() if not mask[0]]
        spatial_ok = space(path)
        if not spatial_ok:
            failed.append("planned_space")
        if np.any(errors > limits):
            failed.append("tracking")
        name = "rollout_%05d" % len(paths)
        arms[name] = dict(
            pdms=float(label[-1] * 100),
            labels=label.tolist(),
            generator_eligible=not failed,
            failed_rules=failed,
            tracking_errors=errors.tolist(),
            planned_space_ok=spatial_ok,
        )
        names.append(name)
        paths.append(path)
        labels.append(label)
        eligible += not failed
        planned_clearance = geometry.clearance(geometry.states(path))
        tracked_clearance = geometry.clearance(tracked)
        spatial_margin = config.get("spatial_margin_m", 0.0)
        clearance_buffer = np.tile(np.minimum(np.arange(41) * 0.1 / 0.5, 1.0), 2) * spatial_margin
        # Keeping the simulated center on-route is a conservative search aid.
        # Final DDC acceptance always uses the official label, unmodified.
        tracked_slack = tracked_clearance - clearance_buffer
        if config.get("tracked_route_rule") == "official-ddc":
            from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
                BBCoordsIndex,
                EgoAreaIndex,
            )

            scorer = evaluator.scorer
            ddc = direction_slack(
                scorer._ego_coords[1, :, BBCoordsIndex.CENTER],
                scorer._ego_areas[1, :, EgoAreaIndex.ONCOMING_TRAFFIC],
                horizon=int(scorer._config.driving_direction_horizon / 0.1),
                threshold_m=scorer._config.driving_direction_compliance_threshold,
            )
            # This changes the optimization aid only. Every retained plan still
            # passes the original full-route check and official DDC == 1.
            tracked_slack = np.r_[tracked_slack[:41], ddc]
        slack = np.r_[
            motion_slack(path, velocity, acceleration, policy),
            planned_clearance - clearance_buffer,
            tracked_slack,
            1 - errors / (limits * 0.98),
        ]
        if config.get("comfort_constraints", False):
            slack = np.r_[slack, comfort_slack(tracked)]
        arms[name]["search_minimum_slack"] = float(slack.min())
        interior = not config.get("comfort_constraints", False) or slack.min() >= -1e-7
        strong += not failed and label[-1] >= config["seed_score_goal"] and interior
        # The progress component offers a gradient even when safety gates fail.
        # It is never substituted for the stored official total score.
        objective = -float(label[2]) + 0.1 * np.maximum(-tracked_clearance, 0).mean()
        memo[key] = (objective, slack)
        return memo[key]

    for parent_id, (parent_name, parent) in enumerate(parents):
        base = parent.astype(float)
        base[:, 2] = np.unwrap(np.r_[0.0, base[:, 2]])[1:]
        spline_mode = config.get("parameterization", "waypoints") == "spline"
        transform = (lambda x: spline_correction(base, x).ravel()) if spline_mode else (lambda x: x)
        width = config.get("position_bound_m", 3.0)
        if spline_mode:
            start = np.zeros(12)
            lower, upper = np.tile([-width, -width, -4.5], 4), np.tile([width, width, 4.5], 4)
        else:
            start = base.ravel()
            lower, upper = (base - [width, width, 0.45]).ravel(), (
                base + [width, width, 0.45]
            ).ravel()
        try:
            result = minimize(
                lambda x: query(transform(x))[0],
                start,
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints=[dict(type="ineq", fun=lambda x: query(transform(x))[1])],
                options=dict(
                    maxiter=config["iterations"],
                    ftol=1e-6,
                    eps=config.get("finite_difference_step", 0.002),
                ),
            )
            query(transform(result.x))
            runs.append(
                dict(
                    parent=parent_name,
                    success=bool(result.success),
                    message=str(result.message),
                    iterations=int(result.nit),
                )
            )
        except SearchStop:
            runs.append(dict(parent=parent_name, stopped="query budget or verified seed goal"))
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
    row = dict(
        token=token,
        record=record,
        arms=arms,
        queries=len(paths),
        eligible=eligible,
        optimizations=runs,
        parent_sources=[n for n, _ in parents],
        no_candidate_score_inherited=True,
    )
    write_json(out / (token + ".json"), row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokens", "policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate-roots", nargs="+", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-queries", type=int, default=2400)
    p.add_argument("--iterations", type=int, default=35)
    p.add_argument("--parents", type=int, default=3)
    p.add_argument("--seed-score-goal", type=float, default=0.96)
    p.add_argument("--parent-selection", choices=("score", "feasibility"), default="score")
    p.add_argument("--comfort-constraints", action="store_true")
    p.add_argument("--spatial-margin-m", type=float, default=0.0)
    p.add_argument("--parameterization", choices=("waypoints", "spline"), default="waypoints")
    p.add_argument("--position-bound-m", type=float, default=3.0)
    p.add_argument("--finite-difference-step", type=float, default=0.002)
    p.add_argument("--tracked-route-rule", choices=("strict", "official-ddc"), default="strict")
    a = p.parse_args()
    import yaml
    from .store import read_json, write_json

    tokens = sorted(a.tokens.read_text().split())
    if len(tokens) != len(set(tokens)) or not tokens:
        raise ValueError("Tokens must be nonempty and unique")
    if (
        min(a.workers, a.max_queries, a.iterations, a.parents) < 1
        or not 0.95 <= a.seed_score_goal <= 1
        or not 0 <= a.spatial_margin_m <= 0.5
        or not 0 < a.position_bound_m <= 10
        or not 0 < a.finite_difference_step <= 0.1
    ):
        raise ValueError("Invalid search budget or score goal")
    records = {
        r["token"]: r for r in map(json.loads, (a.source / "scenes.jsonl").read_text().splitlines())
    }
    config = dict(
        version=1,
        source=str(a.source),
        output=str(a.output),
        tokens=tokens,
        candidate_roots=[str(x) for x in a.candidate_roots],
        max_queries=a.max_queries,
        iterations=a.iterations,
        parents=a.parents,
        seed_score_goal=a.seed_score_goal,
        policy=yaml.safe_load(a.policy.read_text()),
    )
    if a.parent_selection != "score":
        config["parent_selection"] = a.parent_selection
    if a.comfort_constraints:
        config["comfort_constraints"] = True
        config["version"] = 2
    if a.spatial_margin_m:
        config["spatial_margin_m"] = a.spatial_margin_m
        config["version"] = 2
    if (
        a.parameterization != "waypoints"
        or a.position_bound_m != 3.0
        or a.finite_difference_step != 0.002
    ):
        config.update(
            version=3,
            parameterization=a.parameterization,
            position_bound_m=a.position_bound_m,
            finite_difference_step=a.finite_difference_step,
        )
    if a.tracked_route_rule != "strict":
        config.update(version=4, tracked_route_rule=a.tracked_route_rule)
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
        raise RuntimeError("Rollout search contains errors")


if __name__ == "__main__":
    main()
