"""Diagnose stored SimScale shortfalls without changing labels or admission."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import lzma
from pathlib import Path
import pickle
import numpy as np


def inspect_scene(folder):
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

    install_numpy_2_pickle_compat()
    import numpy.core.numeric, sys

    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    from drivesta_feedback.batch.filter_targets import TargetPolicy, tracking_errors
    from drivesta_feedback.batch.physical_sampling import (
        physical_mask,
        score_mask,
        safe_planned_space_checker,
    )
    from drivesta_feedback.batch.batched_official import BatchedOfficial

    folder = Path(folder)
    contract = json.loads((folder / "contract.json").read_text())
    constructor = contract["protocol"]["constructor"]
    policy = TargetPolicy(**constructor["policy"])
    with lzma.open(contract["identity"]["official_cache_path"], "rb") as f:
        cache = pickle.load(f)
    rows = np.load(folder / "candidates.npy", allow_pickle=False)
    with np.load(folder / "construction.npz", allow_pickle=False) as z:
        seeds, tracking = z["seeds"].copy(), z["tracking"].copy()
        alternative, eligible = z["alternative_labels"].copy(), z["eligible"].copy()
    y = rows["labels"]
    all_y, all_alt = [y], [alternative]
    if (folder / "extra_queries.npz").exists():
        with np.load(folder / "extra_queries.npz", allow_pickle=False) as z:
            all_y.append(z["labels"])
            all_alt.append(z["alternative_labels"])
    all_y, all_alt = np.concatenate(all_y), np.concatenate(all_alt)

    def counts(a, b):
        masks = {
            "pdm95": a[:, -1] >= 0.95,
            "nominal_components": score_mask(a, policy),
            "both_environments": score_mask(a, policy) & score_mask(b, policy),
        }
        return {k: int(v.sum()) for k, v in masks.items()}

    limits = [
        policy.max_tracking_ade_m,
        policy.max_tracking_error_m,
        policy.max_tracking_heading_error_rad,
    ]
    tracking_ok = (tracking <= limits).all(1)
    before_track = score_mask(y, policy) & score_mask(alternative, policy)
    check, omitted = safe_planned_space_checker(cache)
    evaluator = BatchedOfficial(cache)
    seed_y, seed_sim = evaluator.score(seeds)
    seed_alt = evaluator.alternative_labels.copy()
    expert = np.asarray(constructor["expert"])
    velocity, acceleration = constructor["initial_velocity"], constructor["initial_acceleration"]
    seed_physics, physics_masks = physical_mask(seeds, expert, policy, velocity, acceleration)
    pose = cache.ego_state.rear_axle
    origin = [pose.x, pose.y, pose.heading]
    seed_track = np.asarray([tracking_errors(t, s, origin) for t, s in zip(seeds, seed_sim)])
    seed_space = np.array([check(t) for t in seeds])
    seed_ok = (
        seed_physics
        & score_mask(seed_y, policy)
        & score_mask(seed_alt, policy)
        & (seed_track <= limits).all(1)
        & seed_space
    )
    meta = json.loads((folder / "seeds.json").read_text())
    # Isolate immutable t=0 geometry: every candidate starts at the same state.
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

    m = cache.drivable_area_map
    road_ids = m.get_indices_of_map_type(
        [
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.CARPARK_AREA,
        ]
    )
    road = shapely.union_all(
        [
            m[m.tokens[i]]
            for i in road_ids
            if m[m.tokens[i]].is_valid and not m[m.tokens[i]].is_empty
        ]
    )
    route = shapely.union_all(
        [
            m[t]
            for t in m.tokens
            if t in cache.route_lane_ids and m[t].is_valid and not m[t].is_empty
        ]
    )
    states = get_trajectory_as_array(
        transform_trajectory(Trajectory(expert), cache.ego_state),
        TrajectorySampling(num_poses=40, interval_length=0.1),
        cache.ego_state.time_point,
    )
    coords = state_array_to_coords_array(states[None], get_pacifica_parameters())[0]
    initial_road = bool(shapely.covers(road, shapely.points(coords[0, :4])).all())
    initial_route = bool(shapely.covers(route, shapely.Point(coords[0, BBCoordsIndex.CENTER])))
    result = dict(
        token=folder.name,
        source=contract["identity"]["source"],
        queries=len(all_y),
        all_query_counts=counts(all_y, all_alt),
        full_counts=counts(y, alternative),
        full_after_tracking=int((before_track & tracking_ok).sum()),
        full_qualified=int(eligible.sum()),
        full_tracking_individual={
            name: int((before_track & (tracking[:, i] <= limits[i])).sum())
            for i, name in enumerate(["ade", "max_position", "max_heading"])
        },
        initial_road=initial_road,
        initial_route=initial_route,
        omitted_geometry=omitted,
        expert_labels=json.loads((folder / "expert.json").read_text())["labels"],
        seeds=[
            dict(
                meta=meta["metadata"][i],
                labels=seed_y[i].tolist(),
                physics=bool(seed_physics[i]),
                physics_failures=[k for k, v in physics_masks.items() if not v[i]],
                tracking=seed_track[i].tolist(),
                space=bool(seed_space[i]),
                eligible=bool(seed_ok[i]),
                selected=i in meta["selected_parents"],
            )
            for i in range(len(seeds))
        ],
        eligible_seed_count=int(seed_ok.sum()),
        selected_eligible_seed_count=sum(bool(seed_ok[i]) for i in meta["selected_parents"]),
        result=json.loads((folder / "RESULT.json").read_text()),
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    folders = [str(a.root / "scenes/train" / t) for t in a.tokens.read_text().split()]
    with ProcessPoolExecutor(a.workers) as pool, (a.output / "diagnostics.jsonl").open("w") as f:
        for result in pool.map(inspect_scene, folders):
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(
                result["token"],
                result["all_query_counts"],
                result["full_after_tracking"],
                result["full_qualified"],
                result["initial_road"],
                result["initial_route"],
                flush=True,
            )


if __name__ == "__main__":
    main()
