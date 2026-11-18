"""Restore source-map precision in isolated caches, with unchanged admission."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import lzma
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from drivesta_feedback.batch.store import atomic, read_json, write_json
from .simscale_recovery import load_cache


def load_scene(identity):
    from drivesta_feedback.paths import bootstrap_navsim

    root = bootstrap_navsim()
    from navsim.common.dataclasses import Scene, SensorConfig

    with open(identity["log_path"], "rb") as f:
        frames = pickle.load(f)
    ids = [i for i, f in enumerate(frames) if f["token"] == identity["token"]]
    if len(ids) != 1 or ids[0] < 3 or len(frames) - ids[0] < 9:
        raise ValueError("Scene identity/horizon mismatch")
    i = ids[0]
    return Scene.from_scene_dict_list(
        frames[i - 3 : i + 9],
        root / "data/sensor_blobs" / identity["source"],
        4,
        8,
        SensorConfig.build_no_sensors(),
    )


def polygon_record(old, source):
    """Report a coordinate substitution, never invent or buffer a boundary."""
    import shapely

    a, b = shapely.get_coordinates(old), shapely.get_coordinates(source)
    same_shape = a.shape == b.shape
    return dict(
        changed=not old.equals_exact(source, 0),
        cached_valid=bool(old.is_valid),
        source_valid=bool(source.is_valid),
        cached_float32_roundtrip=bool(np.array_equal(a, a.astype(np.float32).astype(float))),
        max_coordinate_error_m=float(np.abs(a - b).max()) if same_shape else None,
        float32_source_match=bool(
            same_shape and np.array_equal(a, b.astype(np.float32).astype(float))
        ),
    )


def precise_map(cached, map_api):
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
        PDMDrivableMap,
    )

    geometries, records = [], []
    for token, kind, old in zip(cached.tokens, cached.map_types, cached._geometries):
        obj = map_api.get_map_object(token, kind)
        if obj is None or obj.polygon.is_empty:
            raise ValueError(f"Missing source geometry: {kind.name}/{token}")
        source = obj.polygon
        records.append(dict(token=token, layer=kind.name, **polygon_record(old, source)))
        geometries.append(source)
    restored = PDMDrivableMap(list(cached.tokens), list(cached.map_types), geometries)
    if restored.tokens != cached.tokens or restored.map_types != cached.map_types:
        raise ValueError("Map identity/order changed")
    return restored, records


def initial_space(cache):
    import shapely
    from drivesta_feedback.batch.adaptive_sampling import PreparedSpace
    from drivesta_feedback.batch.filter_targets import MissingPlannedGeometry

    try:
        space = PreparedSpace(cache)
    except MissingPlannedGeometry:
        return dict(road=False, route=False, valid=False, omitted=[], missing=True)
    box = np.asarray(cache.ego_state.car_footprint.geometry.exterior.coords)[:4]
    center = shapely.Point(cache.ego_state.center.x, cache.ego_state.center.y)
    road = bool(shapely.covers(space.road, shapely.points(box)).all())
    route = bool(space.route.covers(center))
    if (road and route) != space(np.zeros((8, 3), np.float32)):
        raise ValueError("Initial geometry and production predicate disagree")
    return dict(
        road=road,
        route=route,
        valid=road and route,
        omitted=space.omitted,
        road_distance_m=float(shapely.distance(shapely.points(box), space.road).max()),
        route_distance_m=float(space.route.distance(center)),
        missing=False,
    )


def check_reuse_intent(result, trace):
    expected = bool(
        trace and result["token"] == trace["token"] and not result["geometry_only"]["valid"]
    )
    applied = result["provenance"]["route_repair"] is not None
    if expected != applied:
        raise ValueError(
            "Route repair intent changed; use a new output directory to preserve existing caches"
        )


def anchor_known_loop(cache, scene, trace):
    """Only the independently traced current-roadblock deletion is corrected."""
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import remove_route_loops
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor

    if set(cache.route_lane_ids) != set(trace["cached_lane_ids"]):
        raise ValueError("Route trace no longer describes this cache")
    center = shapely.Point(cache.ego_state.center.x, cache.ego_state.center.y)
    pose = cache.ego_state.rear_axle
    rear = shapely.Point(pose.x, pose.y)

    def block(k):
        return scene.map_api.get_map_object(k, L.ROADBLOCK) or scene.map_api.get_map_object(
            k, L.ROADBLOCK_CONNECTOR
        )

    choices = []
    for rb in trace["raw_roadblocks"]:
        for lane in block(rb).interior_edges:
            if not lane.polygon.is_valid or not lane.polygon.covers(center):
                continue
            line = lane.baseline_path.linestring
            s = line.project(rear)
            a, b = line.interpolate(max(0, s - 0.5)), line.interpolate(min(line.length, s + 0.5))
            heading = np.arctan2(b.y - a.y, b.x - a.x)
            error = abs(np.arctan2(np.sin(heading - pose.heading), np.cos(heading - pose.heading)))
            if error < np.pi / 4 and line.distance(rear) < 3:
                choices.append(
                    dict(lane=lane.id, roadblock=rb, heading_error_deg=float(np.degrees(error)))
                )
    if len(choices) != 1 or choices[0]["lane"] in cache.route_lane_ids:
        raise ValueError("Route anchoring lacks a unique missing original containing lane")
    anchor = choices[0]
    before = trace["before_loop_removal"]
    if before.count(anchor["roadblock"]) != 1:
        raise ValueError("Ambiguous route anchor")
    suffix = before[before.index(anchor["roadblock"]) :]
    _, after = remove_route_loops([block(k) for k in suffix], suffix)
    if (
        not after
        or after[0] != anchor["roadblock"]
        or not all(a in [v.id for v in block(b).incoming_edges] for a, b in zip(after, after[1:]))
    ):
        raise ValueError("Route suffix is not directed and connected")
    scenario = NavSimScenario(
        scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
    )
    processor = MetricCacheProcessor(None, False)
    inputs, init = processor._get_planner_inputs(scenario)
    planner = processor._pdm_closed
    planner.initialize(init)
    planner._route_roadblock_correction = lambda ego: planner._load_route_dicts(after)
    cache.trajectory = planner.compute_planner_trajectory(inputs)
    cache.centerline = planner._centerline
    cache.route_lane_ids = list(planner._route_lane_dict)
    # Keep the precise same-ID map already restored; no new map extent.
    if anchor["lane"] not in cache.route_lane_ids:
        raise ValueError("Rebuilt route lost anchor")
    return dict(anchor=anchor, connected=True, suffix=after, reference_rebuilt=True)


def repair(token, root, output, trace_path=None):
    folder = root / "scenes/train" / token
    destination = output / "scenes" / token
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "repair.json"
    if result_path.exists():
        result = read_json(result_path)
        if Path(result["cache_path"]).exists():
            check_reuse_intent(result, read_json(trace_path) if trace_path else None)
            return result
    contract = read_json(folder / "contract.json")
    identity = contract["identity"]
    cache = load_cache(identity["official_cache_path"])
    scene = load_scene(identity)
    pose = cache.ego_state.rear_axle
    observed = scene.frames[3]
    delta = np.asarray(observed.ego_status.ego_pose) - [pose.x, pose.y, pose.heading]
    delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
    if abs(delta).max() > 1e-4 or observed.timestamp != cache.ego_state.time_point.time_us:
        raise ValueError("Source frame and current cache pose/time disagree")
    before = initial_space(cache)
    corrected = copy.copy(cache)
    corrected.drivable_area_map, polygons = precise_map(cache.drivable_area_map, scene.map_api)
    geometry_only = initial_space(corrected)
    route = None
    if trace_path:
        trace = read_json(trace_path)
        if token == trace["token"] and not geometry_only["valid"]:
            route = anchor_known_loop(corrected, scene, trace)
    after = initial_space(corrected)
    target = destination / "metric_cache.pkl"
    provenance = dict(
        version="source_geometry_v1",
        source_cache_path=identity["official_cache_path"],
        source_map=scene.scene_metadata.map_name,
        same_map_ids=True,
        source_coordinates_unbuffered=True,
        route_repair=route,
        ego_and_actors_unchanged=True,
    )
    corrected._geometry_repair = provenance
    corrected.file_path = target
    if corrected.ego_state is not cache.ego_state or corrected.observation is not cache.observation:
        raise ValueError("Ego or nominal actors changed")
    if corrected._alternative_observation is not cache._alternative_observation:
        raise ValueError("Alternative actors changed")
    atomic(
        target,
        lambda f: f.write(
            lzma.compress(pickle.dumps(corrected, protocol=pickle.HIGHEST_PROTOCOL), preset=0)
        ),
    )
    row = dict(
        token=token,
        cache_path=str(target),
        provenance=provenance,
        before=before,
        geometry_only=geometry_only,
        after=after,
        changed_polygons=sum(p["changed"] for p in polygons),
        quantized_polygons=sum(p["float32_source_match"] and p["changed"] for p in polygons),
        invalid_cached=sum(not p["cached_valid"] for p in polygons),
        invalid_source=sum(not p["source_valid"] for p in polygons),
        polygons=polygons,
    )
    write_json(result_path, row)
    return row


def worker(args):
    try:
        return repair(*args)
    except Exception:
        return dict(token=args[0], error=traceback.format_exc())


def main():
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "tokens", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--route-trace", type=Path)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    tokens = a.tokens.read_text().split()
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid token list")
    a.output.mkdir(parents=True, exist_ok=True)
    rows = {}
    start = time.time()
    with (
        ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool,
        (a.output / "audit.jsonl").open("a") as stream,
    ):
        futures = [
            pool.submit(worker, (t, a.root.resolve(), a.output.resolve(), a.route_trace))
            for t in tokens
        ]
        for f in as_completed(futures):
            row = f.result()
            rows[row["token"]] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(rows) % 128 == 0:
                print(
                    json.dumps(
                        dict(processed=len(rows), total=len(tokens), elapsed_s=time.time() - start)
                    ),
                    flush=True,
                )
    good = [r for r in rows.values() if "error" not in r]
    summary = dict(
        total=len(tokens),
        errors=[r for r in rows.values() if "error" in r],
        transitions=dict(
            Counter(str(r["before"]["valid"]) + "->" + str(r["after"]["valid"]) for r in good)
        ),
        initially_failed=sum(not r["before"]["valid"] for r in good),
        geometry_fixed=sum(not r["before"]["valid"] and r["geometry_only"]["valid"] for r in good),
        route_fixed=sum(not r["geometry_only"]["valid"] and r["after"]["valid"] for r in good),
        elapsed_s=time.time() - start,
    )
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["errors"]:
        raise RuntimeError("Geometry audit has errors")


if __name__ == "__main__":
    main()
