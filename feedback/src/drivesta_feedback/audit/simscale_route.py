"""Compare initial route membership against original directed map lanes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import traceback

import numpy as np


def inspect(token, root):
    from drivesta_feedback.audit.simscale_recovery import load_cache

    folder = Path(root) / "scenes/train" / token
    contract = json.loads((folder / "contract.json").read_text())
    identity = contract["identity"]
    cache = load_cache(identity["official_cache_path"])
    import shapely
    from navsim.common.dataclasses import Scene, SensorConfig
    from drivesta_feedback.paths import project_root
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.actor_state.state_representation import Point2D

    frames = pickle.load(open(identity["log_path"], "rb"))
    idx = next(i for i, f in enumerate(frames) if f["token"] == token)
    scene = Scene.from_scene_dict_list(
        frames[idx - 3 : idx + 9],
        project_root() / "data/sensor_blobs" / identity["source"],
        4,
        8,
        SensorConfig.build_no_sensors(),
    )
    raw = list(map(str, frames[idx]["roadblock_ids"]))
    center = shapely.Point(cache.ego_state.center.x, cache.ego_state.center.y)
    rear = cache.ego_state.rear_axle
    objects = scene.map_api.get_proximal_map_objects(
        Point2D(rear.x, rear.y), 20.0, [L.LANE, L.LANE_CONNECTOR]
    )
    lanes = []
    for lane in objects[L.LANE] + objects[L.LANE_CONNECTOR]:
        if lane.polygon.distance(center) > 2.0:
            continue
        line = lane.baseline_path.linestring
        s = line.project(shapely.Point(rear.x, rear.y))
        a = line.interpolate(max(0.0, s - 0.5))
        b = line.interpolate(min(line.length, s + 0.5))
        heading = np.arctan2(b.y - a.y, b.x - a.x)
        error = abs(np.arctan2(np.sin(heading - rear.heading), np.cos(heading - rear.heading)))
        lanes.append(
            dict(
                lane_id=lane.id,
                roadblock_id=lane.get_roadblock_id(),
                in_raw_route=str(lane.get_roadblock_id()) in raw,
                in_cache_route=lane.id in cache.route_lane_ids,
                covers_center=bool(lane.polygon.covers(center)),
                center_distance_m=float(lane.polygon.distance(center)),
                heading_error_deg=float(np.degrees(error)),
                valid_geometry=bool(lane.polygon.is_valid),
            )
        )
    missing = [
        x
        for x in lanes
        if x["in_raw_route"]
        and not x["in_cache_route"]
        and x["covers_center"]
        and x["valid_geometry"]
        and x["heading_error_deg"] < 45.0
    ]
    return dict(
        token=token,
        raw_roadblocks=raw,
        nearby_lanes=lanes,
        missing_directed_original_lane=missing,
        source_modified=False,
    )


def safe(token, root):
    try:
        return inspect(token, root)
    except Exception:
        return dict(token=token, error=traceback.format_exc())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "tokens", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    tokens = a.tokens.read_text().split()
    rows = []
    with (
        ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool,
        (a.output / "routes.jsonl").open("w") as f,
    ):
        for result in as_completed([pool.submit(safe, t, str(a.root)) for t in tokens]):
            row = result.result()
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
    from drivesta_feedback.batch.store import write_json

    write_json(
        a.output / "summary.json",
        dict(
            scenes=len(rows),
            errors=[r for r in rows if "error" in r],
            raw_route_missing_lane=sum(bool(r.get("missing_directed_original_lane")) for r in rows),
        ),
    )


if __name__ == "__main__":
    main()
