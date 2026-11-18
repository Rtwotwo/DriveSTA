"""Separate initial spatial contradictions from unresolved trajectory searches."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np


def initial_geometry(cache):
    """Measure current road/route discrepancies using the policy's polygons."""
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.common.dataclasses import Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        state_array_to_coords_array,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex

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
    road = shapely.union_all([dm[dm.tokens[i]] for i in ids])
    route = shapely.union_all([dm[t] for t in dm.tokens if t in cache.route_lane_ids])
    plan = transform_trajectory(Trajectory(np.zeros((8, 3))), cache.ego_state)
    state = get_trajectory_as_array(
        plan, TrajectorySampling(num_poses=40, interval_length=0.1), cache.ego_state.time_point
    )
    coords = state_array_to_coords_array(state[None], get_pacifica_parameters())[0, 0]
    corners = shapely.points(coords[:4])
    center = shapely.Point(coords[BBCoordsIndex.CENTER])
    rear = cache.ego_state.rear_axle
    rear_point = shapely.Point(rear.x, rear.y)
    lane_types = dm.get_indices_of_map_type(
        [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    )
    containing = [dm.tokens[i] for i in lane_types if shapely.covers(dm[dm.tokens[i]], center)]
    return dict(
        road_covers_corners=shapely.covers(road, corners).tolist(),
        corner_distance_outside_road_m=shapely.distance(road, corners).tolist(),
        route_covers_center=bool(shapely.covers(route, center)),
        center_distance_outside_route_m=float(shapely.distance(route, center)),
        route_covers_rear_axle=bool(shapely.covers(route, rear_point)),
        rear_axle_distance_outside_route_m=float(shapely.distance(route, rear_point)),
        center_lane_ids=containing,
        center_lanes_on_route=[t for t in containing if t in cache.route_lane_ids],
        source_route_lane_ids=list(cache.route_lane_ids),
        current_pose=[rear.x, rear.y, rear.heading],
    )


def inspect_one(job):
    record, source = job
    from drivesta_feedback.batch.source import load_scene
    from drivesta_feedback.batch.filter_targets import planned_space_checker

    folder = Path(source) / "scenes" / record["split"] / record["token"]
    contract = json.loads((folder / "contract.json").read_text())
    _, _, cache, identity = load_scene(record, [record["token"]], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Scene identity mismatch")
    # A repeated current pose isolates the spatial condition at t=0. This is
    # only a geometry diagnostic; it is not a driving target or a PDM label.
    current_pose_ok = planned_space_checker(cache)(np.zeros((8, 3), np.float32))
    rows = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
    return dict(
        token=record["token"],
        split=record["split"],
        current_pose_satisfies_spatial_policy=bool(current_pose_ok),
        initial_geometry=initial_geometry(cache),
        original_count=len(rows),
        original_best_pdms=float(rows["labels"][:, -1].max() * 100),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--candidate-roots", nargs="*", type=Path, default=[])
    args = p.parse_args()
    tokens = args.tokens.read_text().split()
    if len(tokens) != len(set(tokens)):
        raise ValueError("Repeated token")
    records = {
        r["token"]: r
        for r in map(json.loads, (args.source / "scenes.jsonl").read_text().splitlines())
    }
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(inspect_one, [(records[t], str(args.source)) for t in tokens]))
    for row in results:
        candidates = []
        failures = {}
        for root in args.candidate_roots:
            path = root / "scenes" / (row["token"] + ".json")
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if data["token"] != row["token"]:
                raise ValueError("Diagnostic token mismatch")
            for name, value in data["arms"].items():
                physical_failure = set(value["failed_rules"]) - {
                    "pdm",
                    "safety_components",
                    "direction_compliance",
                    "tracking",
                }
                if value.get("planned_space_ok") and not physical_failure:
                    candidates.append(dict(source=str(root), name=name, **value))
            for reason, count in data.get("precheck_failures", {}).items():
                failures[reason] = failures.get(reason, 0) + count
        row["best_scored_physical_candidate"] = (
            max(candidates, key=lambda x: x["pdms"]) if candidates else None
        )
        driving_valid = [x for x in candidates if not (set(x["failed_rules"]) - {"pdm"})]
        row["best_driving_valid_candidate"] = (
            max(driving_valid, key=lambda x: x["pdms"]) if driving_valid else None
        )
        row["eligible_seed_found"] = any(not x["failed_rules"] for x in driving_valid)
        row["precheck_failures_across_searches"] = failures
    from drivesta_feedback.batch.store import write_json

    result = dict(
        scenes=len(results),
        current_pose_spatial_failures=sum(
            not x["current_pose_satisfies_spatial_policy"] for x in results
        ),
        all_original_3072_present=all(x["original_count"] == 3072 for x in results),
        records=results,
    )
    write_json(args.output, result)
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "token",
                "split",
                "current_pose_spatial_ok",
                "original_count",
                "best_scored_physical_candidate_pdms",
                "failed_rules",
                "candidate_source",
                "current_center_distance_outside_route_m",
                "best_driving_valid_pdms",
                "eligible_seed_found",
            ]
        )
        for row in results:
            best = row["best_scored_physical_candidate"] or {}
            writer.writerow(
                [
                    row["token"],
                    row["split"],
                    row["current_pose_satisfies_spatial_policy"],
                    row["original_count"],
                    best.get("pdms"),
                    ",".join(best.get("failed_rules", [])),
                    best.get("source"),
                    row["initial_geometry"]["center_distance_outside_route_m"],
                    (row["best_driving_valid_candidate"] or {}).get("pdms"),
                    row["eligible_seed_found"],
                ]
            )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
