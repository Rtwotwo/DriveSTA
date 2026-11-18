"""Audit driving admission separately from representative spacing; no publication."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import lzma
import multiprocessing as mp
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
from drivesta_feedback.batch.filter_targets import (
    TargetPolicy,
    preliminary_rules,
    tracking_errors,
    planned_space_checker,
    wrap,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representatives(paths, labels, eligible, spacing, capacity=256):
    ids = np.flatnonzero(eligible)
    ordered = ids[np.argsort(-labels[ids, 6], kind="stable")]
    selected = []
    for i in ordered:
        if selected:
            distances = np.linalg.norm(paths[selected, :, :2] - paths[i, :, :2], axis=-1).mean(-1)
            # All sweeps still reject exact coordinate duplicates.
            if distances.min() < spacing or np.any(
                np.all(paths[selected] == paths[i], axis=(1, 2))
            ):
                continue
        selected.append(int(i))
        if len(selected) == capacity:
            break
    return selected


def measurements(paths, velocity, acceleration):
    t = np.asarray(paths, float)
    full = np.concatenate([np.zeros((len(t), 1, 3)), t], axis=1)
    v = np.diff(full[..., :2], axis=1) / 0.5
    initial_v = np.broadcast_to(velocity, (len(t), 1, 2))
    a = np.diff(np.concatenate([initial_v, v], axis=1), axis=1) / 0.5
    initial_a = np.broadcast_to(acceleration, (len(t), 1, 2))
    jerk = np.diff(np.concatenate([initial_a, a], axis=1), axis=1) / 0.5
    speed = np.linalg.norm(v, axis=-1)
    yaw = wrap(np.diff(full[..., 2], axis=1))
    heading = full[:, :-1, 2] + yaw / 2
    signed = (v * np.stack([np.cos(heading), np.sin(heading)], -1)).sum(-1)
    return dict(
        speed=speed.max(-1),
        acceleration=np.linalg.norm(a, axis=-1).max(-1),
        jerk=np.linalg.norm(jerk, axis=-1).max(-1),
        initial_velocity=np.linalg.norm(v[:, 0] - initial_v[:, 0], axis=-1),
        yaw_rate=np.where(speed > 1.0, abs(yaw) / 0.5, 0.0).max(-1),
        lateral_acceleration=(speed * abs(yaw) / 0.5).max(-1),
        heading_motion=np.where(
            speed > 1.0, abs(wrap(np.arctan2(v[..., 1], v[..., 0]) - heading)), 0.0
        ).max(-1),
        signed_speed=signed.min(-1),
        backward_distance=np.maximum(-signed, 0).sum(-1) * 0.5,
    )


def geometry(cache):
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

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
    route_ids = [i for i, token in enumerate(m.tokens) if token in cache.route_lane_ids]
    road = shapely.union_all([m[m.tokens[i]] for i in road_ids])
    route = shapely.union_all([m[m.tokens[i]] for i in route_ids])
    if road.is_empty or route.is_empty:
        return None
    # Prepared predicates preserve covers semantics. Reference checks below use
    # the unchanged production checker on the same complete planned trajectories.
    shapely.prepare(road)
    shapely.prepare(route)
    return road, route


def assess_scene(job):
    record, index_meta, out = job
    start = time.monotonic()
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
        PDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex as M,
        WeightedMetricIndex as W,
        BBCoordsIndex,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        state_array_to_coords_array,
    )
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    import shapely

    token = record["token"]
    folder = Path(index_meta["root"]) / "scenes" / record["split"] / token
    signature = {
        name: digest(folder / name) for name in ("candidates.npy", "high_score.npz", "expert.json")
    }
    bank = np.load(folder / "candidates.npy", allow_pickle=False)
    raw_paths = bank["trajectories"]
    expert = json.loads((folder / "expert.json").read_text())
    paths = np.concatenate([raw_paths.astype(float), np.array(expert["trajectory"])[None]], axis=0)
    labels = np.concatenate([bank["labels"], np.array(expert["labels"])[None]], axis=0)
    with np.load(folder / "high_score.npz", allow_pickle=False) as h:
        high_ids = h["query_indices"].copy()
        np.testing.assert_array_equal(h["trajectories"], raw_paths[high_ids])
        np.testing.assert_array_equal(h["labels"], labels[high_ids])
    cache_path = (
        Path(index_meta["official_cache_root"])
        / record["log_name"]
        / "unknown"
        / token
        / "metric_cache.pkl"
    )
    with lzma.open(cache_path, "rb") as f:
        cache = pickle.load(f)
    policy = TargetPolicy(**index_meta["target_policy"])
    dyn = cache.ego_state.dynamic_car_state
    velocity = np.array([dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y])
    acceleration = np.array([dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y])
    masks = preliminary_rules(
        paths, labels, np.array(expert["trajectory"]), policy, velocity, acceleration
    )
    measured = measurements(paths, velocity, acceleration)
    additional = ("tracking_ade", "tracking_max", "tracking_heading", "road_space", "route_space")
    for key in additional:
        masks[key] = np.zeros(len(paths), dtype=bool)
    for key in (
        "tracking_ade",
        "tracking_max",
        "tracking_heading",
        "road_outside_m",
        "route_outside_m",
        "road_initial_outside_m",
        "route_initial_outside_m",
    ):
        measured[key] = np.full(len(paths), np.nan)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling, PDMScorerConfig())
    space = geometry(cache)
    original_space = planned_space_checker(cache) if space is not None else None
    reference = get_trajectory_as_array(cache.trajectory, sampling, cache.ego_state.time_point)
    high = np.flatnonzero(masks["pdm"][:-1])
    checked = np.r_[high, len(raw_paths)]
    verify_ids = set(
        high[np.linspace(0, max(0, len(high) - 1), min(8, len(high)), dtype=int)].tolist()
    )
    verify_ids.add(len(raw_paths))
    max_label_error = 0.0
    space_reference_checks = 0
    pose = cache.ego_state.rear_axle
    for offset in range(0, len(checked), 64):
        ids = checked[offset : offset + 64]
        planned = np.stack(
            [
                get_trajectory_as_array(
                    transform_trajectory(Trajectory(paths[i]), cache.ego_state),
                    sampling,
                    cache.ego_state.time_point,
                )
                for i in ids
            ]
        )
        simulated = simulator.simulate_proposals(
            np.concatenate([reference[None], planned]), cache.ego_state
        )
        scorer.score_proposals(
            simulated,
            cache.observation,
            cache.centerline,
            cache.route_lane_ids,
            cache.drivable_area_map,
        )
        multi = scorer._multi_metrics.prod(axis=0)
        raw = scorer._progress_raw * multi
        denominator = np.maximum(raw[0], raw[1:])
        progress = np.where(
            denominator > scorer._config.progress_distance_threshold,
            raw[1:] / np.maximum(denominator, 1e-12),
            (multi[1:] != 0).astype(float),
        )
        weighted = scorer._weighted_metrics[:, 1:].copy()
        weighted[W.PROGRESS] = progress
        weights = scorer._config.weighted_metrics_array
        total = multi[1:] * (weighted * weights[:, None]).sum(0) / weights.sum()
        actual = np.stack(
            [
                scorer._multi_metrics[M.NO_COLLISION, 1:],
                scorer._multi_metrics[M.DRIVABLE_AREA, 1:],
                progress,
                weighted[W.TTC],
                weighted[W.COMFORTABLE],
                weighted[W.DRIVING_DIRECTION],
                total,
            ],
            -1,
        )
        max_label_error = max(max_label_error, float(abs(actual - labels[ids]).max()))
        if not np.allclose(actual, labels[ids], atol=1e-8, rtol=0):
            raise ValueError("Official stored-label mismatch " + token)
        coords = state_array_to_coords_array(planned, get_pacifica_parameters())
        for j, i in enumerate(ids):
            ade, maximum, heading = tracking_errors(
                paths[i], simulated[j + 1], [pose.x, pose.y, pose.heading]
            )
            for key, value, limit in (
                ("tracking_ade", ade, policy.max_tracking_ade_m),
                ("tracking_max", maximum, policy.max_tracking_error_m),
                ("tracking_heading", heading, policy.max_tracking_heading_error_rad),
            ):
                measured[key][i] = value
                masks[key][i] = value <= limit
            if space is None:
                continue
            road, route = space
            corners = shapely.points(coords[j, :, :4, :].reshape(-1, 2))
            centers = shapely.points(coords[j, :, BBCoordsIndex.CENTER, :])
            road_ok = shapely.covers(road, corners)
            route_ok = shapely.covers(route, centers)
            masks["road_space"][i] = road_ok.all()
            masks["route_space"][i] = route_ok.all()
            road_distance = np.asarray(shapely.distance(road, corners)).reshape(41, 4)
            route_distance = np.asarray(shapely.distance(route, centers))
            measured["road_outside_m"][i] = road_distance.max()
            measured["route_outside_m"][i] = route_distance.max()
            measured["road_initial_outside_m"][i] = road_distance[0].max()
            measured["route_initial_outside_m"][i] = route_distance[0]
            if i in verify_ids:
                if bool(original_space(paths[i])) != bool(road_ok.all() and route_ok.all()):
                    raise ValueError("Geometry audit disagrees with original checker " + token)
                space_reference_checks += 1

    strict = np.logical_and.reduce(list(masks.values()))
    fresh = representatives(raw_paths, labels[:-1], strict[:-1], 0.25)
    if fresh != record["auxiliary_indices"]:
        raise ValueError("Recomputed representative indices differ " + token)
    counts = {}
    for gap in (0.25, 0.10, 0.05, 0.0):
        ids = representatives(raw_paths, labels[:-1], strict[:-1], gap)
        counts[str(gap)] = len(ids)
    first_failure = {}
    survivors = masks["pdm"].copy()
    independent = {}
    exclusive = {}
    for key, mask in masks.items():
        if key == "pdm":
            continue
        first_failure[key] = int((survivors[:-1] & ~mask[:-1]).sum())
        survivors &= mask
        independent[key] = int((masks["pdm"][:-1] & ~mask[:-1]).sum())
        other = np.logical_and.reduce([v for k, v in masks.items() if k != key])
        exclusive[key] = int((other[:-1] & ~mask[:-1]).sum())
    sensitivity = {}
    alternatives = {
        "jerk": measured["jerk"] <= policy.max_jerk_mps3 * 1.1,
        "acceleration": measured["acceleration"] <= policy.max_accel_mps2 * 1.1,
        "initial_velocity": measured["initial_velocity"]
        <= policy.max_initial_velocity_error_mps * 1.1,
        "tracking_ade": measured["tracking_ade"] <= policy.max_tracking_ade_m * 1.1,
        "tracking_max": measured["tracking_max"] <= policy.max_tracking_error_m * 1.1,
        "tracking_heading": measured["tracking_heading"]
        <= policy.max_tracking_heading_error_rad * 1.1,
        "road_space_2cm": measured["road_outside_m"] <= 0.02,
        "route_space_2cm": measured["route_outside_m"] <= 0.02,
    }
    for key, mask in alternatives.items():
        original = key.replace("_2cm", "")
        other = np.logical_and.reduce([v for k, v in masks.items() if k != original])
        sensitivity[key] = int((other[:-1] & mask[:-1] & ~strict[:-1]).sum())
    result = dict(
        token=token,
        split=record["split"],
        log_name=record["log_name"],
        cohort=record["cohort"],
        original_targets=len(record["auxiliary_indices"]),
        full_count=len(raw_paths),
        high_count=len(high),
        safe_count=int(strict[:-1].sum()),
        high256_safe_count=int(strict[high_ids].sum()),
        dedup_counts=counts,
        exclusive_rejections=exclusive,
        independent_rejections=independent,
        first_failure=first_failure,
        sensitivity_additional=sensitivity,
        expert_pdms=float(labels[-1, 6] * 100),
        expert_strict=bool(strict[-1]),
        expert_failed=[key for key, mask in masks.items() if not mask[-1]],
        expert_measurements={key: float(value[-1]) for key, value in measured.items()},
        official_rescored=len(checked),
        max_label_error=max_label_error,
        geometry_reference_checks=space_reference_checks,
        original_index_reproduced=True,
        source_hashes=signature,
        cache_path=str(cache_path),
        source_bank=str(folder),
        seconds=time.monotonic() - start,
        missing_geometry=space is None,
    )
    for name, value in signature.items():
        if digest(folder / name) != value:
            raise ValueError("Source file changed during read-only audit " + token)
    scene_out = Path(out) / "scenes" / token
    scene_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scene_out / "audit.npz",
        trajectories=paths,
        labels=labels,
        rule_names=np.array(list(masks)),
        rules=np.stack(list(masks.values()), -1),
        measurement_names=np.array(list(measured)),
        measurements=np.stack(list(measured.values()), -1),
        high256_ids=high_ids,
        original_representatives=np.array(fresh, dtype=np.int64),
    )
    (scene_out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-cohort", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    groups = {"zero": [], "1_to_15": [], "16_to_63": [], "64_plus": []}
    for r in index["records"]:
        n = len(r["auxiliary_indices"])
        key = "zero" if n == 0 else "1_to_15" if n < 16 else "16_to_63" if n < 64 else "64_plus"
        groups[key].append(r)
    rng = np.random.default_rng(args.seed)
    chosen = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["token"])
        chosen.extend(
            dict(rows[i], cohort=key) for i in rng.choice(len(rows), args.per_cohort, replace=False)
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        seed=args.seed,
        source_index=str(args.index),
        source_index_sha256=digest(args.index),
        scope="equal allocation across four target-count strata; not an unweighted population estimate",
        cohort_population={key: len(v) for key, v in groups.items()},
        records=chosen,
        policy=index["target_policy"],
        production_modified=False,
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Audit output belongs to another sample or source revision")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    meta = {k: index[k] for k in ("root", "official_cache_root", "target_policy")}
    results = []
    errors = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(assess_scene, (r, meta, str(args.output))): r for r in chosen}
        for future in as_completed(futures):
            r = futures[future]
            try:
                value = future.result()
                results.append(value)
                print(
                    json.dumps(
                        {
                            k: value[k]
                            for k in (
                                "token",
                                "cohort",
                                "original_targets",
                                "high_count",
                                "safe_count",
                                "high256_safe_count",
                                "dedup_counts",
                                "seconds",
                            )
                        }
                    ),
                    flush=True,
                )
            except Exception:
                error = dict(token=r["token"], traceback=traceback.format_exc())
                errors.append(error)
                print(json.dumps(error), flush=True)
            (args.output / "progress.json").write_text(
                json.dumps(dict(done=len(results), errors=errors, total=len(chosen)))
            )
    summary = dict(scenes=len(results), errors=errors, cohorts={})
    for cohort in list(groups) + ["all_sampled"]:
        rows = [r for r in results if cohort == "all_sampled" or r["cohort"] == cohort]
        summary["cohorts"][cohort] = dict(
            scenes=len(rows),
            high_count=sum(r["high_count"] for r in rows),
            safe_count=sum(r["safe_count"] for r in rows),
            high256_safe_count=sum(r["high256_safe_count"] for r in rows),
            safe_atleast64=sum(r["safe_count"] >= 64 for r in rows),
            safe_atleast256=sum(r["safe_count"] >= 256 for r in rows),
            dedup_atleast64={
                str(g): sum(r["dedup_counts"][str(g)] >= 64 for r in rows)
                for g in (0.25, 0.10, 0.05, 0.0)
            },
            dedup_totals={
                str(g): sum(r["dedup_counts"][str(g)] for r in rows)
                for g in (0.25, 0.10, 0.05, 0.0)
            },
            expert_strict=sum(r["expert_strict"] for r in rows),
            first_failure=(
                {
                    key: sum(r["first_failure"][key] for r in rows)
                    for key in results[0]["first_failure"]
                }
                if results
                else {}
            ),
            exclusive_rejections=(
                {
                    key: sum(r["exclusive_rejections"][key] for r in rows)
                    for key in results[0]["exclusive_rejections"]
                }
                if results
                else {}
            ),
            sensitivity_additional=(
                {
                    key: sum(r["sensitivity_additional"][key] for r in rows)
                    for key in results[0]["sensitivity_additional"]
                }
                if results
                else {}
            ),
        )
    summary.update(
        official_rescored=sum(r["official_rescored"] for r in results),
        max_label_error=max((r["max_label_error"] for r in results), default=None),
        geometry_reference_checks=sum(r["geometry_reference_checks"] for r in results),
        source_unchanged=not errors,
        results=sorted(results, key=lambda r: (r["cohort"], r["token"])),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
