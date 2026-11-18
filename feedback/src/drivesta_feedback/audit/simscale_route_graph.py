"""Audit directed lane membership at the immutable pose and along source GT."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from .simscale_geometry import load_scene, initial_space
from .simscale_recovery import load_cache
from drivesta_feedback.batch.store import read_json, write_json


def heading_error(lane, pose):
    from shapely import Point

    line = lane.baseline_path.linestring
    point = Point(pose[:2])
    s = line.project(point)
    a, b = line.interpolate(max(0.0, s - 0.5)), line.interpolate(min(line.length, s + 0.5))
    heading = np.arctan2(b.y - a.y, b.x - a.x)
    return float(abs(np.arctan2(np.sin(heading - pose[2]), np.cos(heading - pose[2])))), float(
        line.distance(point)
    )


def nearby_lanes(map_api, pose, center_offset=1.461):
    import shapely
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L

    x, y, h = pose
    center = shapely.Point(x + center_offset * np.cos(h), y + center_offset * np.sin(h))
    objects = map_api.get_proximal_map_objects(Point2D(x, y), 12.0, [L.LANE, L.LANE_CONNECTOR])
    result = []
    for lane in objects[L.LANE] + objects[L.LANE_CONNECTOR]:
        if lane.polygon.is_empty or not lane.polygon.is_valid:
            continue
        distance = lane.polygon.distance(center)
        if distance > 2.0:
            continue
        error, rear_distance = heading_error(lane, pose)
        result.append(
            dict(
                lane_id=lane.id,
                roadblock_id=lane.get_roadblock_id(),
                covers_center=bool(lane.polygon.covers(center)),
                center_distance_m=float(distance),
                heading_error_deg=float(np.degrees(error)),
                rear_distance_m=rear_distance,
            )
        )
    return result


def shortest_connection(start, target, successors, max_depth=30):
    """Directed BFS. Multiple shortest routes are reported as ambiguous."""
    queue = deque([(start, [start])])
    depths = {start: 0}
    visits = {start: 1}
    answers = []
    best = None
    while queue:
        node, path = queue.popleft()
        depth = len(path) - 1
        if best is not None and depth > best:
            break
        if node == target:
            answers.append(path)
            best = depth
            if len(answers) > 1:
                break
            continue
        if depth >= max_depth:
            continue
        for nxt in sorted(set(successors(node))):
            if nxt in path:
                continue
            if depths.get(nxt, depth + 1) < depth + 1:
                continue
            if depths.get(nxt) == depth + 1 and visits.get(nxt, 0) >= 2:
                continue
            visits[nxt] = visits.get(nxt, 0) + 1 if depths.get(nxt) == depth + 1 else 1
            depths[nxt] = depth + 1
            queue.append((nxt, path + [nxt]))
    return answers


def inspect(token, root, cache_root):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    folder = Path(root) / "scenes/train" / token
    identity = read_json(folder / "contract.json")["identity"]
    cache = load_cache(Path(cache_root) / "scenes" / token / "metric_cache.pkl")
    scene = load_scene(identity)
    with open(identity["log_path"], "rb") as f:
        frames = pickle.load(f)
    index = next(i for i, f in enumerate(frames) if f["token"] == token)
    raw = list(dict.fromkeys(map(str, frames[index]["roadblock_ids"])))
    pose = np.array(
        [
            cache.ego_state.rear_axle.x,
            cache.ego_state.rear_axle.y,
            cache.ego_state.rear_axle.heading,
        ]
    )
    gt = np.asarray(identity["expert_gt"])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    future = np.column_stack(
        (gt[:, :2] @ np.array([[c, s], [-s, c]]) + pose[:2], gt[:, 2] + pose[2])
    )
    offset = get_pacifica_parameters().rear_axle_to_center
    lanes = [nearby_lanes(scene.map_api, p, offset) for p in np.vstack([pose, future])]
    directed = [
        [
            x
            for x in group
            if x["covers_center"] and x["heading_error_deg"] < 45 and x["rear_distance_m"] < 3
        ]
        for group in lanes
    ]
    current_blocks = sorted(set(x["roadblock_id"] for x in directed[0]))
    cached = set(cache.route_lane_ids)
    source_pose = np.array(frames[index]["ego2global_translation"][:2])
    matrix_pose = np.asarray(frames[index]["ego2global"])[:2, 3]
    targets = []
    for step, group in enumerate(directed[1:], 1):
        for x in group:
            if x["roadblock_id"] in raw and x["roadblock_id"] not in [
                a["roadblock"] for a in targets
            ]:
                targets.append(dict(step=step, roadblock=x["roadblock_id"]))
    blocks = {}

    def block(k):
        if k not in blocks:
            blocks[k] = scene.map_api.get_map_object(
                k, L.ROADBLOCK
            ) or scene.map_api.get_map_object(k, L.ROADBLOCK_CONNECTOR)
        return blocks[k]

    bridges = []
    for start in current_blocks:
        for target in targets[:3]:
            paths = shortest_connection(
                start, target["roadblock"], lambda k: [x.id for x in block(k).outgoing_edges]
            )
            bridges.append(dict(start=start, target=target, paths=paths))
    return dict(
        token=token,
        initial=initial_space(cache),
        raw_roadblocks=raw,
        cached_route_lane_ids=cache.route_lane_ids,
        nearby_at_start=lanes[0],
        current_directed_blocks=current_blocks,
        gt_directed_membership=directed,
        gt_cached_membership=[
            any(x["covers_center"] and x["lane_id"] in cached for x in group) for group in lanes
        ],
        bridges=bridges,
        source_pose_delta_m=float(np.linalg.norm(source_pose - matrix_pose)),
        no_containing_directed_lane=not bool(current_blocks),
        source_map=scene.scene_metadata.map_name,
    )


def safe(args):
    try:
        return inspect(*args)
    except Exception:
        return dict(token=args[0], error=traceback.format_exc())


def main():
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "cache-root", "tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    tokens = a.tokens.read_text().split()
    assert len(tokens) == len(set(tokens))
    journal = a.output / "audit.jsonl"
    rows = {}
    if journal.exists():
        rows = {z["token"]: z for z in map(json.loads, journal.read_text().splitlines())}
    pending = [t for t in tokens if t not in rows or "error" in rows[t]]
    start = time.time()
    with (
        ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool,
        journal.open("a") as f,
    ):
        futures = [pool.submit(safe, (t, str(a.root), str(a.cache_root))) for t in pending]
        for future in as_completed(futures):
            z = future.result()
            rows[z["token"]] = z
            f.write(json.dumps(z) + "\n")
            f.flush()
            if len(rows) % 128 == 0:
                print(
                    json.dumps(
                        dict(processed=len(rows), total=len(tokens), elapsed_s=time.time() - start)
                    ),
                    flush=True,
                )
    good = [z for z in rows.values() if "error" not in z]
    summary = dict(
        scenes=len(rows),
        errors=[z for z in rows.values() if "error" in z],
        current_directed_blocks=dict(Counter(len(z["current_directed_blocks"]) for z in good)),
        no_containing_directed_lane=sum(z["no_containing_directed_lane"] for z in good),
        unique_bridge=sum(any(len(b["paths"]) == 1 for b in z["bridges"]) for z in good),
        elapsed_s=time.time() - start,
    )
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["errors"]:
        raise RuntimeError("Source audit incomplete")


if __name__ == "__main__":
    main()
