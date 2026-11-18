"""Audit initial route conflicts against source maps and a fresh official cache."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import lzma
import os
from pathlib import Path
import pickle

import numpy as np


def inspect(job):
    record, source, output = job
    from drivesta_feedback.batch.source import load_scene, Evaluator
    from drivesta_feedback.audit.recovery_remaining import initial_geometry
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    token = record["token"]
    contract = json.loads(
        (Path(source) / "scenes" / record["split"] / token / "contract.json").read_text()
    )
    scene, expert, cache, identity = load_scene(record, [token], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source identity mismatch")
    scenario = NavSimScenario(
        scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
    )
    raw_blocks = scenario.get_route_roadblock_ids()
    raw_lanes = []
    for block_id in raw_blocks:
        block = scene.map_api.get_map_object(block_id, SemanticMapLayer.ROADBLOCK)
        block = block or scene.map_api.get_map_object(
            block_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        if block is not None:
            raw_lanes.extend(edge.id for edge in block.interior_edges)
    old_geometry = initial_geometry(cache)
    containing = []
    for lane_id in old_geometry["center_lane_ids"]:
        lane = scene.map_api.get_map_object(lane_id, SemanticMapLayer.LANE)
        lane = lane or scene.map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
        containing.append(
            dict(
                lane_id=lane_id,
                roadblock_id=lane.get_roadblock_id(),
                in_raw_route=lane_id in raw_lanes,
                in_cached_route=lane_id in cache.route_lane_ids,
            )
        )
    # A separate location avoids changing the source or training caches.
    processor = MetricCacheProcessor(
        cache_path=str(Path(output) / "fresh_cache"), force_feature_computation=True
    )
    entry = processor.compute_metric_cache(scenario)
    with lzma.open(entry.file_name, "rb") as f:
        fresh = pickle.load(f)
    old_labels, fresh_labels = Evaluator(cache).score(expert), Evaluator(fresh).score(expert)
    return dict(
        token=token,
        record=record,
        raw_route_roadblocks=raw_blocks,
        containing_lanes=containing,
        old_geometry=old_geometry,
        fresh_geometry=initial_geometry(fresh),
        fresh_route_ids_match=set(cache.route_lane_ids) == set(fresh.route_lane_ids),
        source_expert_labels=old_labels.tolist(),
        fresh_expert_labels=fresh_labels.tolist(),
        max_label_error=float(abs(old_labels - fresh_labels).max()),
        fresh_cache=str(entry.file_name),
        source_modified=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    tokens = a.tokens.read_text().split()
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate tokens")
    a.output.mkdir(parents=True, exist_ok=True)
    records = {
        r["token"]: r for r in map(json.loads, (a.source / "scenes.jsonl").read_text().splitlines())
    }
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(inspect, [(records[t], str(a.source), str(a.output)) for t in tokens]))
    result = dict(
        scenes=len(rows),
        records=rows,
        fresh_route_matches=sum(r["fresh_route_ids_match"] for r in rows),
        fresh_initial_route_passes=sum(r["fresh_geometry"]["route_covers_center"] for r in rows),
        max_expert_label_error=max(r["max_label_error"] for r in rows),
    )
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
