"""Source-supported route anchoring; no pose changes or polygon dilation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import lzma
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import traceback

import numpy as np

from .simscale_geometry import load_scene, initial_space
from .simscale_recovery import load_cache
from .simscale_route_graph import inspect, shortest_connection
from drivesta_feedback.batch.store import atomic, read_json, write_json, checksum


def bounded_connections(start, target, successors, max_edges):
    """Find up to two simple directed connections, including unequal lengths."""
    queue = [(start, [start])]
    found = []
    while queue:
        node, path = queue.pop()
        if node == target:
            found.append(path)
            if len(found) == 2:
                return found
        elif len(path) - 1 < max_edges:
            for nxt in sorted(set(successors(node))):
                if nxt not in path:
                    queue.append((nxt, path + [nxt]))
    return found


def complete_source_suffix(raw, successors, supports, max_edges=6):
    """Restore only a unique witnessed connector between ordered source blocks."""
    result = raw[:1]
    for a, b in zip(raw, raw[1:]):
        if b in successors(a):
            result.append(b)
            continue
        paths = bounded_connections(a, b, successors, max_edges)
        if len(paths) != 1 or not all(supports(k) for k in paths[0][1:-1]):
            return None
        result.extend(paths[0][1:])
    return result


def route_choice(
    audit, successors, supports, max_edges=6, complete_connectors=False, connector_supports=None
):
    """Return a unique directed route justified before any candidate scoring."""
    if audit["initial"]["valid"]:
        return None, "initial_route_already_valid"
    anchors = audit["current_directed_blocks"]
    if len(anchors) != 1:
        return None, "no_unique_containing_directed_roadblock"
    anchor = anchors[0]
    raw = audit["raw_roadblocks"]
    # First observed future-GT return to the intended original route.
    targets = []
    for group in audit["gt_directed_membership"][1:]:
        targets = sorted(set(x["roadblock_id"] for x in group if x["roadblock_id"] in raw))
        if targets:
            break
    if anchor in raw:
        targets = [anchor]
    if len(targets) != 1:
        return None, "ambiguous_or_absent_original_route_target"
    target = targets[0]
    paths = bounded_connections(anchor, target, successors, max_edges)
    if len(paths) != 1:
        return None, "no_unique_short_directed_connection"
    prefix = paths[0]
    if not all(supports(k) for k in prefix):
        return None, "connection_not_supported_by_source_gt"
    suffix = raw[raw.index(target) :]
    if complete_connectors:
        suffix = complete_source_suffix(
            suffix, successors, connector_supports or supports, max_edges
        )
        if suffix is None:
            return None, "source_suffix_without_unique_witnessed_connection"
    route = prefix + suffix[1:]
    if len(set(route)) != len(route):
        return None, "route_repeats_roadblock"
    if not all(b in successors(a) for a, b in zip(route, route[1:])):
        return None, "original_suffix_disconnected"
    return route, "unique_source_supported_connection"


def repair(
    token, root, cache_root, output, audit=None, complete_connectors=False, witnessed_prefix=False
):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    import shapely

    destination = Path(output) / "scenes" / token
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(cache_root) / "scenes" / token / "metric_cache.pkl"
    record_path = destination / "repair.json"
    identity = read_json(Path(root) / "scenes/train" / token / "contract.json")["identity"]
    if record_path.exists():
        row = read_json(record_path)
        if bool(row.get("complete_connectors", False)) != complete_connectors:
            raise ValueError("Connector repair intent changed; use an internal new output")
        if bool(row.get("witnessed_prefix", False)) != witnessed_prefix:
            raise ValueError("GT-witness repair intent changed; use an internal new output")
        if (
            row["source_digest"] != checksum(source)
            or row["provenance"]["source_cache_path"] != identity["official_cache_path"]
        ):
            raise ValueError("Source changed since directed audit")
        if checksum(Path(row["cache_path"])) != row["cache_digest"]:
            raise ValueError("Repaired cache changed")
        return row
    audit = audit or inspect(token, root, cache_root)
    cache = load_cache(source)
    corrected = copy.copy(cache)
    route = None
    reason = "initial_route_already_valid"
    if (not audit["initial"]["valid"] or witnessed_prefix) and (
        len(audit["current_directed_blocks"]) == 1
        or (witnessed_prefix and audit["current_directed_blocks"])
    ):
        scene = load_scene(identity)
        blocks = {}

        def block(k):
            if k not in blocks:
                blocks[k] = scene.map_api.get_map_object(
                    k, L.ROADBLOCK
                ) or scene.map_api.get_map_object(k, L.ROADBLOCK_CONNECTOR)
            if blocks[k] is None:
                raise ValueError("Missing original roadblock " + k)
            return blocks[k]

        def successors(k):
            return [v.id for v in block(k).outgoing_edges]

        pose = cache.ego_state.rear_axle
        c, s = np.cos(pose.heading), np.sin(pose.heading)
        gt = np.vstack((np.zeros((1, 3)), np.asarray(identity["expert_gt"])))
        xy = gt[:, :2] @ np.array([[c, s], [-s, c]]) + [pose.x, pose.y]
        witness = shapely.LineString(xy)

        def supports(k):
            return block(k).polygon.distance(witness) <= 3.0

        connector_witnesses = {}

        def connector_supports(k):
            old_lanes = sorted(
                set(x.id for x in block(k).interior_edges) & set(cache.route_lane_ids)
            )
            gt_steps = [
                i
                for i, group in enumerate(audit["gt_directed_membership"])
                if any(x["roadblock_id"] == k for x in group)
            ]
            connector_witnesses[k] = dict(
                existing_route_lanes=old_lanes, directed_gt_steps=gt_steps
            )
            return bool(old_lanes or gt_steps)

        if witnessed_prefix:
            from .simscale_route_witness import witnessed_choice

            route, reason, _ = witnessed_choice(audit, successors, connector_supports)
            if route and route[0] in audit["raw_roadblocks"]:
                route, reason = None, "no_added_gt_witnessed_prefix"
        else:
            route, reason = route_choice(
                audit,
                successors,
                supports,
                complete_connectors=complete_connectors,
                connector_supports=connector_supports,
            )
        if route:
            scenario = NavSimScenario(
                scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
            )
            processor = MetricCacheProcessor(None, False)
            inputs, init = processor._get_planner_inputs(scenario)
            planner = processor._pdm_closed
            planner.initialize(init)
            planner._route_roadblock_correction = lambda ego: planner._load_route_dicts(route)
            corrected.trajectory = planner.compute_planner_trajectory(inputs)
            corrected.centerline = planner._centerline
            corrected.route_lane_ids = list(planner._route_lane_dict)
    elif not audit["initial"]["valid"]:
        reason = "no_unique_containing_directed_roadblock"
    provenance = dict(
        version="source_gt_witnessed_route_v1" if witnessed_prefix else "source_directed_route_v1",
        source_cache_path=identity["official_cache_path"],
        precise_cache_path=str(source),
        source_coordinates_unbuffered=True,
        same_map_ids=True,
        ego_and_actors_unchanged=True,
        route_repair=(
            dict(
                roadblocks=route,
                reason=reason,
                reference_rebuilt=True,
                selected_without_scores=True,
            )
            if route
            else None
        ),
    )
    if route and (complete_connectors or witnessed_prefix):
        provenance["route_repair"]["connector_witnesses"] = connector_witnesses
    if route and witnessed_prefix:
        provenance["route_repair"]["gt_directed_membership"] = audit["gt_directed_membership"]
        provenance["route_repair"]["raw_roadblocks"] = audit["raw_roadblocks"]
        provenance["route_repair"][
            "route_inference"
        ] = "unique_gt_witnessed_prefix_rejoining_original_suffix"
    corrected._geometry_repair = provenance
    for name in ("ego_state", "observation", "_alternative_observation", "drivable_area_map"):
        if getattr(corrected, name) is not getattr(cache, name):
            raise ValueError("Immutable scene changed: " + name)
    after = initial_space(corrected)
    if route and not after["valid"]:
        raise ValueError("Directed repair did not satisfy unchanged initial-space check")
    target = destination / "metric_cache.pkl"
    corrected.file_path = target
    if route:
        atomic(
            target,
            lambda f: f.write(
                lzma.compress(pickle.dumps(corrected, protocol=pickle.HIGHEST_PROTOCOL), preset=0)
            ),
        )
    else:
        # Reuse the immutable precision cache without copying its actor arrays.
        target = source
    row = dict(
        token=token,
        cache_path=str(target),
        provenance=provenance,
        before=audit["initial"],
        after=after,
        complete_connectors=complete_connectors,
        witnessed_prefix=witnessed_prefix,
        reason=reason,
        source_digest=checksum(source),
        cache_digest=checksum(target),
        route_repaired=bool(route),
    )
    write_json(record_path, row)
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
    for name in ("root", "cache-root", "tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--audit", type=Path)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--complete-connectors", action="store_true")
    p.add_argument("--witnessed-prefix", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    tokens = a.tokens.read_text().split()
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid tokens")
    audits = (
        {z["token"]: z for z in map(json.loads, a.audit.read_text().splitlines())}
        if a.audit
        else {}
    )
    rows = []
    with (
        ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool,
        (a.output / "audit.jsonl").open("a") as stream,
    ):
        tasks = [
            pool.submit(
                worker,
                (
                    t,
                    a.root,
                    a.cache_root,
                    a.output,
                    audits.get(t),
                    a.complete_connectors,
                    a.witnessed_prefix,
                ),
            )
            for t in tokens
        ]
        for f in as_completed(tasks):
            z = f.result()
            rows.append(z)
            stream.write(json.dumps(z) + "\n")
            stream.flush()
    summary = dict(
        scenes=len(rows),
        errors=[z for z in rows if "error" in z],
        repaired=sum(z.get("route_repaired", False) for z in rows),
        initial_valid=sum(z.get("after", {}).get("valid", False) for z in rows),
    )
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["errors"]:
        raise RuntimeError("Directed route repair errors")


if __name__ == "__main__":
    main()
