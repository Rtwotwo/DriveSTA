"""Build a variable-size generator-target view without editing scored banks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import lzma
import multiprocessing as mp
from pathlib import Path
import pickle

import numpy as np


class MissingPlannedGeometry(ValueError):
    """The cache cannot support the requested spatial target verification."""


@dataclass(frozen=True)
class TargetPolicy:
    min_pdm: float = 0.95
    max_targets: int = 8
    min_ade_m: float = 0.25
    max_track_queries: int = 64
    max_speed_mps: float = 25.0
    max_accel_mps2: float = 6.0
    max_jerk_mps3: float = 10.0
    max_yaw_rate_rps: float = 1.0
    max_lateral_accel_mps2: float = 4.0
    min_signed_speed_mps: float = -0.2
    max_heading_motion_error_rad: float = 0.35
    moving_speed_mps: float = 1.0
    max_initial_velocity_error_mps: float = 2.0
    max_tracking_ade_m: float = 1.0
    max_tracking_error_m: float = 2.0
    max_tracking_heading_error_rad: float = 0.35
    check_planned_space: bool = False
    max_backward_distance_m: float = 0.01

    def __post_init__(self):
        if not 0 <= self.min_pdm <= 1 or not 1 <= self.max_targets <= self.max_track_queries:
            raise ValueError("Invalid target count, query budget or score threshold")
        for name, value in asdict(self).items():
            if name not in ("min_signed_speed_mps",) and (not np.isfinite(value) or value < 0):
                raise ValueError(name)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def preliminary_rules(
    trajectories, labels, expert, policy, initial_velocity=None, initial_acceleration=None
):
    """All masks refer to stored plans; DDC and PDM refer to official rollouts."""
    t, y, gt = np.asarray(trajectories, float), np.asarray(labels, float), np.asarray(expert, float)
    if t.ndim != 3 or t.shape[1:] != (8, 3) or y.shape != (len(t), 7) or gt.shape != (8, 3):
        raise ValueError("Unexpected bank or expert shape")
    if not np.isfinite(t).all() or not np.isfinite(y).all() or not np.isfinite(gt).all():
        raise ValueError("Nonfinite source data")
    if (y < 0).any() or (y > 1).any():
        raise ValueError("Official labels must be in 0..1")
    expected = y[:, 0] * y[:, 1] * (5 * y[:, 2] + 5 * y[:, 3] + 2 * y[:, 4]) / 12
    if not np.allclose(expected, y[:, -1], atol=1e-8, rtol=0):
        raise ValueError("Stored score/component mismatch")
    full = np.concatenate([np.zeros((len(t), 1, 3)), t], axis=1)
    velocity = np.diff(full[..., :2], axis=1) / 0.5
    speed = np.linalg.norm(velocity, axis=-1)
    initial_v = (
        velocity[:, :1]
        if initial_velocity is None
        else np.broadcast_to(np.asarray(initial_velocity, float), (len(t), 1, 2))
    )
    acceleration = np.diff(np.concatenate([initial_v, velocity], axis=1), axis=1) / 0.5
    initial_a = (
        acceleration[:, :1]
        if initial_acceleration is None
        else np.broadcast_to(np.asarray(initial_acceleration, float), (len(t), 1, 2))
    )
    jerk = np.diff(np.concatenate([initial_a, acceleration], axis=1), axis=1) / 0.5
    yaw_delta = wrap(np.diff(full[..., 2], axis=1))
    yaw_rate = abs(yaw_delta) / 0.5
    mid_heading = full[:, :-1, 2] + yaw_delta / 2
    direction = np.stack([np.cos(mid_heading), np.sin(mid_heading)], -1)
    signed_speed = (velocity * direction).sum(-1)
    moving = speed > policy.moving_speed_mps
    heading_error = abs(wrap(np.arctan2(velocity[..., 1], velocity[..., 0]) - mid_heading))
    return {
        "pdm": y[:, -1] >= policy.min_pdm,
        "safety_components": np.all(y[:, [0, 1, 3, 4]] == 1.0, axis=1),
        "direction_compliance": y[:, 5] == 1.0,
        "forward_motion": signed_speed.min(-1) >= policy.min_signed_speed_mps,
        "backward_distance": np.maximum(-signed_speed, 0).sum(-1) * 0.5
        <= policy.max_backward_distance_m,
        "speed": speed.max(-1) <= policy.max_speed_mps,
        "acceleration": np.linalg.norm(acceleration, axis=-1).max(-1) <= policy.max_accel_mps2,
        "jerk": np.linalg.norm(jerk, axis=-1).max(-1) <= policy.max_jerk_mps3,
        "yaw_rate": np.where(moving, yaw_rate, 0.0).max(-1) <= policy.max_yaw_rate_rps,
        "lateral_acceleration": (speed * yaw_rate).max(-1) <= policy.max_lateral_accel_mps2,
        "heading_motion": np.where(moving, heading_error, 0.0).max(-1)
        <= policy.max_heading_motion_error_rad,
        "initial_velocity": np.linalg.norm(velocity[:, 0] - initial_v[:, 0], axis=-1)
        <= policy.max_initial_velocity_error_mps,
    }


def select_targets(
    trajectories, labels, expert, policy, verify, initial_velocity=None, initial_acceleration=None
):
    """Descending official score, geometric deduplication, then rollout checks.

    verify(index) returns an independently rescored label and tracking errors.
    No duplicated padding or score assignment to the original expert is allowed.
    """
    masks = preliminary_rules(
        trajectories, labels, expert, policy, initial_velocity, initial_acceleration
    )
    remaining = np.ones(len(trajectories), dtype=bool)
    funnel = {"input": int(remaining.sum())}
    for name, mask in masks.items():
        remaining &= mask
        funnel[name] = int(remaining.sum())
    eligible = np.flatnonzero(remaining)
    ordered = eligible[np.argsort(-labels[eligible, -1], kind="stable")]
    selected, checked = [], []
    deduplicated = 0
    for i in ordered:
        if len(selected) == policy.max_targets or len(checked) == policy.max_track_queries:
            break
        if selected:
            distances = np.linalg.norm(
                trajectories[selected, :, :2] - trajectories[i, :, :2], axis=-1
            ).mean(-1)
            if distances.min() < policy.min_ade_m:
                deduplicated += 1
                continue
        verification = verify(int(i))
        labels_again, ade, max_error, heading_error = verification[:4]
        if policy.check_planned_space and len(verification) < 5:
            raise ValueError("Planned-space verification is required by the target policy")
        spatial_ok = bool(verification[4]) if len(verification) > 4 else True
        if not np.allclose(labels_again, labels[i], atol=1e-8, rtol=0):
            raise ValueError(f"Official label replay differs at query {i}")
        if not np.isfinite([ade, max_error, heading_error]).all():
            raise ValueError("Nonfinite tracking errors")
        accepted = (
            spatial_ok
            and ade <= policy.max_tracking_ade_m
            and max_error <= policy.max_tracking_error_m
            and heading_error <= policy.max_tracking_heading_error_rad
        )
        checked.append(
            dict(
                query_index=int(i),
                tracking_ade_m=float(ade),
                tracking_max_error_m=float(max_error),
                tracking_max_heading_error_rad=float(heading_error),
                planned_space_ok=spatial_ok,
                accepted=bool(accepted),
            )
        )
        if accepted:
            selected.append(int(i))
    return dict(
        query_indices=selected,
        target_count=len(selected),
        preserve_expert=True,
        expert_only=len(selected) == 0,
        funnel=funnel,
        checked=checked,
        skipped_near_duplicates=deduplicated,
        verification_budget_exhausted=len(checked) == policy.max_track_queries
        and len(selected) < policy.max_targets,
    )


def read_target_view(bank_folder, selection):
    """Training input contract: expert plus variable-length, scored auxiliaries."""
    folder = Path(bank_folder)
    rows = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
    indices = np.asarray(selection["query_indices"], dtype=np.int64)
    if (
        indices.ndim != 1
        or len(np.unique(indices)) != len(indices)
        or (indices < 0).any()
        or (indices >= len(rows)).any()
    ):
        raise ValueError("Invalid source indices")
    if "selected_labels" in selection:
        expected = np.asarray(selection["selected_labels"], float).reshape(-1, 7)
        if not np.array_equal(expected, rows["labels"][indices]):
            raise ValueError("Source bank changed after filtering")
    expert = np.asarray(
        json.loads((folder / "expert.json").read_text())["trajectory"], dtype=np.float32
    )
    return dict(
        expert=expert,
        auxiliary_trajectories=np.array(rows["trajectories"][indices]),
        auxiliary_labels=np.array(rows["labels"][indices]),
        auxiliary_count=len(indices),
    )


def tracking_errors(trajectory, simulated_states, initial_pose):
    """Compare planned and tracked rear-axle poses at all official 0.1s times."""
    trajectory = np.asarray(trajectory, float)
    sim = np.asarray(simulated_states, float)
    pose = np.asarray(initial_pose, float)
    if (
        trajectory.shape != (8, 3)
        or sim.ndim != 2
        or sim.shape[0] != 41
        or sim.shape[1] < 3
        or pose.shape != (3,)
    ):
        raise ValueError("Unexpected trajectory, simulation or initial pose shape")
    if not all(np.isfinite(x).all() for x in (trajectory, sim, pose)):
        raise ValueError("Nonfinite tracking input")
    c, s = np.cos(pose[2]), np.sin(pose[2])
    local = (sim[:, :2] - pose[:2]) @ np.array([[c, -s], [s, c]])
    plan = np.concatenate([np.zeros((1, 2)), trajectory[:, :2]], axis=0)
    planned = np.stack(
        [np.interp(np.arange(41) * 0.1, np.arange(9) * 0.5, plan[:, d]) for d in range(2)], -1
    )
    error = np.linalg.norm(local - planned, axis=-1)
    planned_heading = np.interp(
        np.arange(41) * 0.1, np.arange(9) * 0.5, np.unwrap(np.r_[0.0, trajectory[:, 2]])
    )
    heading_error = abs(wrap(sim[:, 2] - pose[2] - planned_heading))
    return float(error[1:].mean()), float(error.max()), float(heading_error.max())


def filter_scene(job):
    record, bank_root, cache_root, policy_dict = job
    policy = TargetPolicy(**policy_dict)
    folder = Path(bank_root) / "scenes" / record["split"] / record["token"]
    rows = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
    gt = np.asarray(json.loads((folder / "expert.json").read_text())["trajectory"])
    from .source import Evaluator

    path = Path(cache_root) / record["log_name"] / "unknown" / record["token"] / "metric_cache.pkl"
    with lzma.open(path, "rb") as f:
        cache = pickle.load(f)
    evaluator = Evaluator(cache)
    missing_geometry = False
    try:
        space_checker = planned_space_checker(cache) if policy.check_planned_space else None
    except MissingPlannedGeometry:
        space_checker = None
        missing_geometry = True

    def verify(i):
        label, sim = evaluator.score(rows["trajectories"][i], details=True)
        pose = evaluator.cache.ego_state.rear_axle
        spatial_ok = space_checker(rows["trajectories"][i]) if space_checker else True
        return (
            (label,)
            + tracking_errors(rows["trajectories"][i], sim, [pose.x, pose.y, pose.heading])
            + (spatial_ok,)
        )

    dynamic = cache.ego_state.dynamic_car_state
    initial_v = [dynamic.rear_axle_velocity_2d.x, dynamic.rear_axle_velocity_2d.y]
    initial_a = [dynamic.rear_axle_acceleration_2d.x, dynamic.rear_axle_acceleration_2d.y]
    if missing_geometry:
        masks = preliminary_rules(
            rows["trajectories"], rows["labels"], gt, policy, initial_v, initial_a
        )
        remaining = np.ones(len(rows), dtype=bool)
        funnel = {"input": len(rows)}
        for name, mask in masks.items():
            remaining &= mask
            funnel[name] = int(remaining.sum())
        result = dict(
            query_indices=[],
            target_count=0,
            preserve_expert=True,
            expert_only=True,
            funnel=funnel,
            checked=[],
            skipped_near_duplicates=0,
            verification_budget_exhausted=False,
            missing_planned_geometry=True,
        )
    else:
        result = select_targets(
            rows["trajectories"], rows["labels"], gt, policy, verify, initial_v, initial_a
        )
    result.update(record)
    result["source_bank"] = str(folder)
    result["selected_labels"] = rows["labels"][result["query_indices"]].tolist()
    result["selection_status"] = (
        "expert_only_missing_geometry"
        if missing_geometry
        else (
            "verification_budget_exhausted"
            if result["verification_budget_exhausted"]
            else "auxiliary_available" if result["target_count"] else "expert_only"
        )
    )
    loaded = read_target_view(folder, result)
    assert loaded["auxiliary_count"] == result["target_count"]
    return result


def planned_space_checker(cache):
    """Require planned vehicle corners in drivable union and center on route."""
    import shapely
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.common.dataclasses import Trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        state_array_to_coords_array,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex

    driving_map = cache.drivable_area_map
    road_ids = driving_map.get_indices_of_map_type(
        [
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.CARPARK_AREA,
        ]
    )
    route_ids = [i for i, t in enumerate(driving_map.tokens) if t in cache.route_lane_ids]
    road = shapely.union_all([driving_map[driving_map.tokens[i]] for i in road_ids])
    route = shapely.union_all([driving_map[driving_map.tokens[i]] for i in route_ids])
    if road.is_empty or route.is_empty:
        raise MissingPlannedGeometry("Missing drivable/route geometry")
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)

    def check(trajectory):
        global_plan = transform_trajectory(Trajectory(trajectory), cache.ego_state)
        states = get_trajectory_as_array(global_plan, sampling, cache.ego_state.time_point)
        coords = state_array_to_coords_array(states[None], get_pacifica_parameters())[0]
        corners = coords[:, :4, :]
        center = coords[:, BBCoordsIndex.CENTER, :]
        return bool(
            shapely.covers(road, shapely.points(corners.reshape(-1, 2))).all()
            and shapely.covers(route, shapely.points(center)).all()
        )

    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--tokens", nargs="+")
    parser.add_argument("--max-track-queries", type=int)
    args = parser.parse_args()
    import yaml

    policy = TargetPolicy(**yaml.safe_load(args.policy.read_text()))
    if args.max_track_queries is not None:
        policy = replace(policy, max_track_queries=args.max_track_queries)
    index = json.loads(args.index.read_text())
    records = sorted(index["records"], key=lambda r: r["token"])
    if args.count < 1 or args.count > len(records):
        raise ValueError("Invalid scene count")
    rng = np.random.default_rng(args.seed)
    selected = [records[i] for i in sorted(rng.choice(len(records), args.count, replace=False))]
    if args.tokens:
        selected = [r for r in records if r["token"] in set(args.tokens)]
        if set(r["token"] for r in selected) != set(args.tokens):
            raise ValueError("Unknown token")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    save = dict(
        policy=asdict(policy),
        seed=args.seed,
        source_index=str(args.index),
        scope="random existing trainval scenes; no new model training",
    )
    (args.output / "config.json").write_text(json.dumps(save, indent=2) + "\n")
    jobs = [(r, index["root"], str(args.cache_root), asdict(policy)) for r in selected]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        with (args.output / "targets.jsonl").open("w") as f:
            for result in pool.map(filter_scene, jobs):
                results.append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()
                if len(results) % 16 == 0:
                    print("filtered", len(results), "/", len(selected), flush=True)
    summary = dict(
        scenes=len(results),
        with_auxiliary=sum(r["target_count"] > 0 for r in results),
        expert_only=sum(r["expert_only"] for r in results),
        targets=sum(r["target_count"] for r in results),
        budget_exhausted=sum(r["verification_budget_exhausted"] for r in results),
        target_count_histogram={
            str(n): sum(r["target_count"] == n for r in results)
            for n in range(policy.max_targets + 1)
        },
        funnel={k: sum(r["funnel"][k] for r in results) for k in results[0]["funnel"]},
        tracking_checked=sum(len(r["checked"]) for r in results),
        tracking_rejected=sum(not c["accepted"] for r in results for c in r["checked"]),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
