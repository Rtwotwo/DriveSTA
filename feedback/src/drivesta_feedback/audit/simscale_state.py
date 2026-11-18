"""Read-only checks of synthetic poses, transforms, map identity and source GT."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
import inspect as python_inspect
import multiprocessing as mp
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from .simscale_geometry import load_scene, initial_space
from .simscale_recovery import load_cache
from .simscale_route_graph import heading_error
from drivesta_feedback.batch.store import read_json, write_json


def finite(value, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError("Invalid numeric source shape or nonfinite value")
    return a


def wrapped(value):
    return np.arctan2(np.sin(value), np.cos(value))


def quaternion_yaw(q):
    q = finite(q, (4,))
    if np.linalg.norm(q) == 0:
        raise ValueError("Zero quaternion")
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z)))


def pose_errors(a, b):
    a, b = finite(a), finite(b)
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("Pose shape mismatch")
    return dict(
        xy_m=float(np.abs(a[..., :2] - b[..., :2]).max()),
        heading_rad=float(np.abs(wrapped(a[..., 2] - b[..., 2])).max()),
    )


def fingerprint(path):
    p = Path(path)
    before = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during digest")
    return dict(
        path=str(p.resolve()), size=after.st_size, mtime_ns=after.st_mtime_ns, sha256=h.hexdigest()
    )


def quaternion_matrix(q):
    """Independent Hamilton wxyz rotation; compare to redundant source matrices."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("Invalid wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def local_xy(global_xy, origin):
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return (np.asarray(global_xy) - origin[:2]) @ np.array([[c, -s], [s, c]])


def can_dynamic_errors(can, dynamics):
    can, dynamics = finite(can, (18,)), finite(dynamics, (4,))
    return (
        float(np.abs(can[10:12] - dynamics[:2]).max()),
        float(np.abs(can[7:9] - dynamics[2:]).max()),
    )


def numeric_maxima(rows):
    excluded = {
        "source_index",
        "matched_actors",
        "source_actor_count",
        "cached_actor_count",
        "source_directed_anchor_count",
        "center_heading_anchor_count",
    }
    keys = sorted(
        {
            k
            for row in rows
            for k, value in row.items()
            if (value is None or isinstance(value, (float, int)))
            and not isinstance(value, bool)
            and k not in excluded
        }
    )
    return {
        k: max((abs(row[k]) for row in rows if row.get(k) is not None), default=None) for k in keys
    }


def inspect(token, root):
    import shapely
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L

    folder = Path(root) / "scenes/train" / token
    identity = read_json(folder / "contract.json")["identity"]
    if identity["token"] != token:
        raise ValueError("Contract token mismatch")
    inputs = [
        folder / "contract.json",
        Path(identity["log_path"]),
        Path(identity["official_cache_path"]),
    ]
    sources = [fingerprint(p) for p in inputs]
    with open(identity["log_path"], "rb") as f:
        frames = pickle.load(f)
    indices = [i for i, frame in enumerate(frames) if frame["token"] == token]
    if len(indices) != 1:
        raise ValueError("Token does not select exactly one source frame")
    ix = indices[0]
    frame = frames[ix]
    source = frames[ix - 3 : ix + 9]
    if ix < 3 or len(source) != 12:
        raise ValueError("Missing source history or future")
    scene = load_scene(identity)
    if (
        scene.scene_metadata.initial_token != token
        or scene.scene_metadata.log_name != frame["log_name"]
        or frame["log_name"] != identity["log_name"]
    ):
        raise ValueError("Loaded scene identity mismatch")
    cache = load_cache(identity["official_cache_path"])
    source_poses = np.array(
        [
            [
                *finite(f["ego2global_translation"], (3,))[:2],
                quaternion_yaw(f["ego2global_rotation"]),
            ]
            for f in source
        ]
    )
    pose = source_poses[3]
    scene_poses = finite([f.ego_status.ego_pose for f in scene.frames], (12, 3))
    cache_pose = np.array(
        [
            cache.ego_state.rear_axle.x,
            cache.ego_state.rear_axle.y,
            cache.ego_state.rear_axle.heading,
        ]
    )
    gt_source = np.asarray(scene.get_future_trajectory().poses)
    raw_gt = finite(identity["raw_expert_gt"], (8, 3)) if "raw_expert_gt" in identity else None
    direct_xy = local_xy(np.asarray([f["ego2global_translation"][:2] for f in source[4:]]), pose)
    transform_errors = []
    translation_errors = []
    can_errors = []
    can_rotation_errors = []
    can_velocity_errors = []
    can_acceleration_errors = []
    yaw_deltas = []
    q_norm_errors = []
    rotation_structure = []
    matrix_bottom_rows = []
    for fr in source:
        matrix = finite(fr["ego2global"], (4, 4))
        q = finite(fr["ego2global_rotation"], (4,))
        can = finite(fr["can_bus"], (18,))
        dynamics = finite(fr["ego_dynamic_state"], (4,))
        transform_errors.append(
            float(np.max(np.abs(quaternion_matrix(fr["ego2global_rotation"]) - matrix[:3, :3])))
        )
        translation_errors.append(
            float(np.max(np.abs(np.asarray(fr["ego2global_translation"]) - matrix[:3, 3])))
        )
        can_errors.append(float(np.max(np.abs(np.asarray(fr["can_bus"])[:3] - matrix[:3, 3]))))
        can_rotation_errors.append(
            float(np.abs(quaternion_matrix(can[3:7]) - quaternion_matrix(q)).max())
        )
        # OpenScene CAN schema: quaternion3:7, acceleration7:10, velocity10:13.
        velocity_error, acceleration_error = can_dynamic_errors(can, dynamics)
        can_velocity_errors.append(velocity_error)
        can_acceleration_errors.append(acceleration_error)
        yaw_deltas.append(
            float(abs(wrapped(np.arctan2(matrix[1, 0], matrix[0, 0]) - quaternion_yaw(q))))
        )
        q_norm_errors.append(float(abs(np.linalg.norm(q) - 1)))
        rotation_structure.append(
            float(
                max(
                    np.abs(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3)).max(),
                    abs(np.linalg.det(matrix[:3, :3]) - 1),
                )
            )
        )
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
            matrix_bottom_rows.append(dict(token=fr["token"], bottom_row=matrix[3].tolist()))
    source_maps = {f["map_location"] for f in source}
    offset = get_pacifica_parameters().rear_axle_to_center
    center = shapely.Point(pose[0] + offset * np.cos(pose[2]), pose[1] + offset * np.sin(pose[2]))
    cached_center = shapely.Point(cache.ego_state.center.x, cache.ego_state.center.y)
    rear = shapely.Point(pose[:2])
    layers = [L.LANE, L.LANE_CONNECTOR]
    proximal = scene.map_api.get_proximal_map_objects(Point2D(*pose[:2]), 30.0, layers)
    lanes = []
    route_ids = set(cache.route_lane_ids)
    for layer in layers:
        for lane in proximal[layer]:
            polygon = lane.polygon
            if polygon.is_empty or polygon.distance(center) > 4:
                continue
            error, distance = heading_error(lane, pose)
            center_error, center_distance = heading_error(
                lane, np.array([center.x, center.y, pose[2]])
            )
            lanes.append(
                dict(
                    id=lane.id,
                    layer=layer.name,
                    roadblock=lane.get_roadblock_id(),
                    valid=bool(polygon.is_valid),
                    covers_center=bool(polygon.covers(center)),
                    covers_rear=bool(polygon.covers(rear)),
                    distance_m=float(polygon.distance(center)),
                    heading_error_deg=float(np.degrees(error)),
                    baseline_distance_m=distance,
                    center_heading_error_deg=float(np.degrees(center_error)),
                    center_baseline_distance_m=center_distance,
                    in_cache_route=lane.id in route_ids,
                )
            )
    valid_center = [x for x in lanes if x["valid"] and x["covers_center"]]
    anchors = [
        x for x in valid_center if x["heading_error_deg"] < 45 and x["baseline_distance_m"] < 3
    ]
    center_anchors = [
        x
        for x in valid_center
        if x["center_heading_error_deg"] < 45
        and x["baseline_distance_m"] < 3
        and x["center_baseline_distance_m"] < 3
    ]
    invalid_center = [x for x in lanes if not x["valid"] and x["covers_center"]]
    # Compare same-ID source geometries close to the immutable center.
    map_errors = []
    missing_objects = []
    cache_ids = set(cache.drivable_area_map.tokens)
    for key, layer, polygon in zip(
        cache.drivable_area_map.tokens,
        cache.drivable_area_map.map_types,
        cache.drivable_area_map._geometries,
    ):
        if polygon.distance(center) > 6:
            continue
        obj = scene.map_api.get_map_object(key, layer)
        if obj is None:
            missing_objects.append([key, layer.name])
            continue
        hausdorff = float(polygon.hausdorff_distance(obj.polygon))
        map_errors.append(
            dict(
                id=key,
                layer=layer.name,
                hausdorff_m=hausdorff if np.isfinite(hausdorff) else None,
                cached_covers=bool(polygon.covers(center)),
                source_covers=bool(obj.polygon.covers(center)),
                source_valid=bool(obj.polygon.is_valid),
            )
        )
    actors = []
    boxes = finite(frame["anns"]["gt_boxes"])
    ids = frame["anns"]["track_tokens"]
    occupancy = cache.observation[0]
    if len(ids) != len(boxes) or len(ids) != len(set(map(str, ids))):
        raise ValueError("Source actor cardinality or ID uniqueness mismatch")
    if len(occupancy.tokens) != len(occupancy._geometries) or len(occupancy.tokens) != len(
        set(map(str, occupancy.tokens))
    ):
        raise ValueError("Cache actor cardinality or ID uniqueness mismatch")
    cached_centers = {
        str(t): np.asarray([g.centroid.x, g.centroid.y])
        for t, g in zip(occupancy.tokens, occupancy._geometries)
    }
    for xy in cached_centers.values():
        finite(xy, (2,))
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, s], [-s, c]])
    for t, box in zip(ids, boxes):
        if str(t) in cached_centers:
            xy = box[:2] @ rotation + pose[:2]
            actors.append(float(np.linalg.norm(xy - cached_centers[str(t)])))
    history_xy = local_xy(np.asarray([f["ego2global_translation"][:2] for f in source[:4]]), pose)
    initial = initial_space(cache)
    dynamic = cache.ego_state.dynamic_car_state
    source_velocity = finite(frame["ego_dynamic_state"][:2], (2,))
    cached_velocity = np.array([dynamic.rear_axle_velocity_2d.x, dynamic.rear_axle_velocity_2d.y])
    cached_accel = finite(
        [dynamic.rear_axle_acceleration_2d.x, dynamic.rear_axle_acceleration_2d.y], (2,)
    )
    times = finite([f["timestamp"] for f in source])
    intervals = np.diff(times) / 1e6
    if (intervals <= 0).any():
        raise ValueError("Non-increasing source timestamps")
    future_times = (times[3:] - times[3]) / 1e6
    local_heading = wrapped(source_poses[4:, 2] - pose[2])
    direct_gt = np.column_stack([direct_xy, local_heading])
    roundtrip = np.column_stack(
        [gt_source[:, :2] @ rotation + pose[:2], wrapped(gt_source[:, 2] + pose[2])]
    )
    comparisons = {}
    for name, a, b in [
        ("cache_pose", pose, cache_pose),
        ("contract_pose", pose, identity["ego_pose"]),
        ("source_scene_pose", source_poses, scene_poses),
        ("independent_gt", gt_source, direct_gt),
        ("global_roundtrip", roundtrip, source_poses[4:]),
        ("planned_reference", gt_source, identity["expert_gt"]),
    ]:
        comparisons.update({name + "_" + k: v for k, v in pose_errors(a, b).items()})
    comparisons.update(
        {"raw_gt_" + k: v for k, v in pose_errors(gt_source, raw_gt).items()}
        if raw_gt is not None
        else dict(raw_gt_xy_m=None, raw_gt_heading_rad=None)
    )
    for p, record in zip(inputs, sources):
        stat = p.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError("Source changed during audit")
    return dict(
        token=token,
        source=identity["source"],
        map=scene.scene_metadata.map_name,
        source_index=ix,
        **comparisons,
        source_fingerprints=sources,
        source_map_names_consistent=len(source_maps) == 1
        and scene.scene_metadata.map_name == frame["map_location"],
        source_rear_axle_to_center_m=offset,
        source_cache_center_error_m=float(center.distance(cached_center)),
        cache_timestamp_error_us=int(cache.ego_state.time_point.time_us - frame["timestamp"]),
        contract_timestamp_error_us=int(identity["timestamp_us"] - frame["timestamp"]),
        cache_velocity_max_error=float(np.abs(source_velocity - cached_velocity).max()),
        cache_acceleration_max_error=float(
            np.abs(finite(frame["ego_dynamic_state"][2:], (2,)) - cached_accel).max()
        ),
        rotation_matrix_max_error=max(transform_errors),
        translation_max_error=max(translation_errors),
        can_translation_max_error=max(can_errors),
        can_rotation_max_error=max(can_rotation_errors),
        can_velocity_max_error_mps=max(can_velocity_errors),
        can_acceleration_max_error_mps2=max(can_acceleration_errors),
        matrix_projected_yaw_convention_delta_rad=max(yaw_deltas),
        quaternion_norm_error=max(q_norm_errors),
        rotation_structure_error=max(rotation_structure),
        nonstandard_homogeneous_bottom_rows=matrix_bottom_rows,
        frame_intervals_s=intervals.tolist(),
        max_interval_error_s=float(abs(intervals - 0.5).max()),
        contract_future_time_error_s=float(
            np.abs(finite(identity["future_times_s"], (9,)) - future_times).max()
        ),
        actor_center_max_error_m=max(actors) if actors else None,
        matched_actors=len(actors),
        source_actor_count=len(ids),
        cached_actor_count=len(cached_centers),
        unmatched_source_actor_ids=sorted(set(map(str, ids)) - set(cached_centers)),
        unmatched_cached_actor_ids=sorted(set(cached_centers) - set(map(str, ids))),
        cache_initial_valid=initial["valid"],
        initial_space=initial,
        source_lanes=lanes,
        source_directed_anchor_count=len(anchors),
        invalid_source_polygon_contains_center=invalid_center,
        center_heading_anchor_count=len(center_anchors),
        local_map_checks=map_errors,
        missing_map_objects=missing_objects,
        nearby_source_lanes_missing_from_cache=[x["id"] for x in lanes if x["id"] not in cache_ids],
        source_velocity=source_velocity.tolist(),
        history_local_xy=history_xy.tolist(),
        gt_local_xy=gt_source[:, :2].tolist(),
        source_log=identity["log_path"],
    )


def worker(args):
    try:
        return inspect(*args)
    except Exception:
        return dict(token=args[0], error=traceback.format_exc())


def main():
    from drivesta_feedback.paths import bootstrap_navsim

    repo = bootstrap_navsim()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    tokens = a.tokens.read_text().split()
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate tokens")
    a.output.mkdir(parents=True, exist_ok=True)
    journal = a.output / "audit.jsonl"
    if journal.exists():
        raise ValueError(
            "Use a fresh output directory; audit rows cannot be reused across source changes"
        )
    rows = {}
    from navsim.common.dataclasses import Scene
    import os

    maproot = Path(os.environ["NUPLAN_MAPS_ROOT"]).resolve()
    metadata = dict(
        version=3,
        root=str(a.root.resolve()),
        tokens=tokens,
        map_root=str(maproot),
        implementations=[
            fingerprint(p)
            for p in sorted(
                {
                    Path(__file__),
                    Path(python_inspect.getfile(Scene)),
                    Path(python_inspect.getfile(load_scene)),
                    Path(python_inspect.getfile(load_cache)),
                    Path(python_inspect.getfile(heading_error)),
                    repo / "navsim/planning/scenario_builder/navsim_scenario.py",
                }
            )
        ],
        map_files=[fingerprint(p) for p in sorted(maproot.rglob("*.gpkg"))],
    )
    write_json(a.output / "provenance.json", metadata)
    start = time.time()
    with (
        ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool,
        journal.open("a") as f,
    ):
        futures = [pool.submit(worker, (t, str(a.root))) for t in tokens]
        for future in as_completed(futures):
            x = future.result()
            rows[x["token"]] = x
            f.write(json.dumps(x, allow_nan=False) + "\n")
            f.flush()
            if len(rows) % 64 == 0:
                print(json.dumps(dict(processed=len(rows), total=len(tokens))), flush=True)
    good = [x for x in rows.values() if "error" not in x]
    summary = dict(
        scenes=len(rows),
        errors=[x for x in rows.values() if "error" in x],
        maxima=numeric_maxima(good),
        source_map_names_consistent=sum(x["source_map_names_consistent"] for x in good),
        raw_gt_available=sum(x["raw_gt_xy_m"] is not None for x in good),
        actor_comparisons_available=sum(x["matched_actors"] > 0 for x in good),
        directed_anchor_scenes=sum(x["source_directed_anchor_count"] > 0 for x in good),
        center_heading_anchor_scenes=sum(x["center_heading_anchor_count"] > 0 for x in good),
        invalid_polygon_contains_scenes=sum(
            bool(x["invalid_source_polygon_contains_center"]) for x in good
        ),
        missing_source_objects=sum(bool(x["missing_map_objects"]) for x in good),
        maps=dict(Counter(x["map"] for x in good)),
        elapsed_s=time.time() - start,
    )
    for record in metadata["implementations"] + metadata["map_files"]:
        stat = Path(record["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError("Code or map changed during audit")
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["errors"]:
        raise RuntimeError("Source audit incomplete")


if __name__ == "__main__":
    main()
