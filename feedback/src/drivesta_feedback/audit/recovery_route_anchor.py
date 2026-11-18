"""Test current-lane route anchoring without editing source routes or labels."""

import argparse
from copy import copy
import json
from pathlib import Path
import numpy as np


def choose_anchor(lanes, max_heading_deg=45.0, max_baseline_distance_m=3.0):
    """A containing polygon alone does not identify a directed driving lane."""
    eligible = [
        x
        for x in lanes
        if x["in_original_route"]
        and x["heading_error_deg"] < max_heading_deg
        and x["baseline_distance_m"] < max_baseline_distance_m
    ]
    return (
        min(
            eligible, key=lambda x: (x["baseline_distance_m"], x["heading_error_deg"], x["lane_id"])
        )
        if eligible
        else None
    )


def render_comparison(cache, expert, result, output):
    """Display unchanged source polygons for the two alternative route lists."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L

    dm = cache.drivable_area_map
    pose = cache.ego_state.rear_axle
    origin = np.array([pose.x, pose.y])
    c, s = np.cos(pose.heading), np.sin(pose.heading)
    rotation = np.array([[c, -s], [s, c]])

    def local(xy):
        return (np.asarray(xy) - origin) @ rotation

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, key in zip(axes, ("cached", "directed_anchor_suffix")):
        variant = result["variants"][key]
        ids = set(variant["route_lane_ids"])
        for i in dm.get_indices_of_map_type([L.LANE, L.LANE_CONNECTOR]):
            name = dm.tokens[i]
            polygon = dm[name]
            if polygon.distance(cache.ego_state.car_footprint.geometry) > 25:
                continue
            polygons = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
            for p in polygons:
                ax.add_patch(
                    Polygon(
                        local(p.exterior.coords),
                        facecolor="#62a8d1" if name in ids else "#dddddd",
                        edgecolor="#79828a",
                        lw=0.4,
                        alpha=0.35,
                    )
                )
        box = local(cache.ego_state.car_footprint.geometry.exterior.coords)
        ax.plot(box[:, 0], box[:, 1], "k-", lw=1.4)
        center = local([[cache.ego_state.center.x, cache.ego_state.center.y]])[0]
        ax.scatter(
            *center, c="#c52c36", marker="x", s=60, zorder=10, label="Initial vehicle center"
        )
        ax.arrow(0, 0, 4, 0, width=0.06, head_width=0.5, color="black", length_includes_head=True)
        ax.plot(
            np.r_[0, expert[:, 0]],
            np.r_[0, expert[:, 1]],
            "--",
            c="#b52f86",
            lw=1.8,
            label="Recorded GT",
        )
        ax.set_title(
            ("Cached route" if key == "cached" else "Directed, connected route suffix")
            + "\nInitial center outside by %.2f m" % variant["center_distance_m"]
        )
        ax.set(xlim=(-9, 28), ylim=(-18, 18), xlabel="Forward x (m)", ylabel="Left y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(result["token"] + " | Route membership audit", fontsize=14)
    fig.text(
        0.08,
        0.025,
        "Blue: tested route polygons. Grey: other lanes. Source geometry and initial state are unchanged.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output / "scenes" / (result["token"] + ".png"), dpi=160)
    plt.close(fig)


def inspect(record, source, trace_root, output):
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import remove_route_loops
    from drivesta_feedback.batch.source import load_scene, Evaluator
    from drivesta_feedback.batch.recovery_seeds import resample_recorded
    from drivesta_feedback.batch.filter_targets import planned_space_checker
    from drivesta_feedback.batch.store import atomic, write_json

    token = record["token"]
    contract = json.loads(
        (source / "scenes" / record["split"] / token / "contract.json").read_text()
    )
    scene, expert, cache, identity = load_scene(record, [token], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source identity changed")
    trace = json.loads((trace_root / "scenes" / (token + ".json")).read_text())
    raw_ids = trace["raw_route_ids"]
    before = trace["correction_trace"]["before_loop_removal"]
    original_route = tuple(cache.route_lane_ids)
    dm = cache.drivable_area_map
    center = shapely.Point(cache.ego_state.center.x, cache.ego_state.center.y)
    rear = cache.ego_state.rear_axle
    ego = shapely.Point(rear.x, rear.y)

    def block(k):
        return scene.map_api.get_map_object(k, L.ROADBLOCK) or scene.map_api.get_map_object(
            k, L.ROADBLOCK_CONNECTOR
        )

    def lane_ids(ids):
        return [lane.id for k in ids for lane in block(k).interior_edges]

    lanes = []
    for k in dict.fromkeys(lane_ids(raw_ids) + trace["initial_geometry"]["center_lane_ids"]):
        lane = scene.map_api.get_map_object(k, L.LANE) or scene.map_api.get_map_object(
            k, L.LANE_CONNECTOR
        )
        if lane.polygon.distance(center) > 15:
            continue
        line = lane.baseline_path.linestring
        s = line.project(ego)
        a, b = line.interpolate(max(0, s - 0.5)), line.interpolate(min(line.length, s + 0.5))
        heading = np.arctan2(b.y - a.y, b.x - a.x)
        error = abs(np.arctan2(np.sin(heading - rear.heading), np.cos(heading - rear.heading)))
        lanes.append(
            dict(
                lane_id=k,
                roadblock_id=lane.get_roadblock_id(),
                in_original_route=lane.get_roadblock_id() in raw_ids,
                covers_center=bool(lane.polygon.covers(center)),
                baseline_distance_m=float(line.distance(ego)),
                heading_error_deg=float(np.degrees(error)),
            )
        )
    anchor = choose_anchor(lanes)
    # Historical poses can identify the intended directed lane when the
    # current pose lies in a map seam. They do not waive current containment.
    history = []
    current_index = scene.scene_metadata.num_history_frames - 1
    for frame in scene.frames[:current_index]:
        pose = np.asarray(frame.ego_status.ego_pose, float)
        point = shapely.Point(pose[:2])
        choices = []
        for candidate in lanes:
            lane = scene.map_api.get_map_object(
                candidate["lane_id"], L.LANE
            ) or scene.map_api.get_map_object(candidate["lane_id"], L.LANE_CONNECTOR)
            line = lane.baseline_path.linestring
            s = line.project(point)
            a, b = line.interpolate(max(0, s - 0.5)), line.interpolate(min(line.length, s + 0.5))
            heading = np.arctan2(b.y - a.y, b.x - a.x)
            error = abs(np.arctan2(np.sin(heading - pose[2]), np.cos(heading - pose[2])))
            choices.append(
                dict(
                    candidate,
                    baseline_distance_m=float(line.distance(point)),
                    heading_error_deg=float(np.degrees(error)),
                )
            )
        history.append(
            dict(
                time_s=(frame.timestamp - scene.frames[current_index].timestamp) / 1e6,
                anchor=choose_anchor(choices),
            )
        )
    anchor_source = "current"
    if anchor is None:
        supported = [h["anchor"] for h in history if h["anchor"] is not None]
        if supported and len({a["roadblock_id"] for a in supported}) == 1 and len(supported) >= 2:
            anchor = min(supported, key=lambda x: x["baseline_distance_m"])
            anchor_source = "consistent_history"
    variants = {
        "cached": list(original_route),
        "original_route": lane_ids(raw_ids),
        "before_loop_cut": lane_ids(before),
    }
    anchor_trace = None
    if anchor:
        # Retain official graph-link repairs, then remove the obsolete prefix.
        # Starting from raw ids would reintroduce missing roadblock links.
        if before.count(anchor["roadblock_id"]) != 1:
            raise ValueError("Ambiguous anchor occurrence in repaired route")
        suffix = before[before.index(anchor["roadblock_id"]) :]
        objects = [block(k) for k in suffix]
        objects, after = remove_route_loops(objects, suffix)
        variants["directed_anchor_suffix"] = lane_ids(after)
        anchor_trace = dict(
            anchor=anchor,
            source=anchor_source,
            before=suffix,
            after=list(after),
            preserves_anchor=anchor["roadblock_id"] in after,
            connected=all(
                a in [z.id for z in block(b).incoming_edges] for a, b in zip(after, after[1:])
            ),
        )
    start = scene.scene_metadata.num_history_frames - 1
    times = (
        np.array([f.timestamp for f in scene.frames[start:]], float) - scene.frames[start].timestamp
    ) / 1e6
    future = scene.get_future_trajectory(num_trajectory_frames=len(times) - 1).poses
    timed = resample_recorded(np.vstack((np.zeros((1, 3)), future)), times)
    tests = {"recorded_expert": expert.astype(np.float32), "timestamp_expert": timed}
    # Nearby saved plans are an additional diagnostic, never accepted by a new map.
    proposal_path = trace_root.parent / "proposal25_v2/scenes" / (token + ".npz")
    if proposal_path.exists():
        with np.load(proposal_path, allow_pickle=False) as z:
            for i in np.argsort(-z["labels"][:, -1])[:3]:
                tests["proposal_" + str(i)] = z["trajectories"][i].copy()
    details = {}
    all_labels = []
    keys = []
    all_paths = []
    for name, ids in variants.items():
        # Counterfactual isolates route membership. Reference/centerline stay
        # fixed, so these labels are diagnostic and never production labels.
        c = copy(cache)
        c.route_lane_ids = list(dict.fromkeys(ids))
        available = [k for k in c.route_lane_ids if k in dm.tokens]
        polygon = shapely.union_all([dm[k] for k in available])
        evaluator = Evaluator(c)
        space = planned_space_checker(c)
        paths = {}
        for kind, path in tests.items():
            label, tracked = evaluator.score(path, details=True)
            paths[kind] = dict(
                labels=label.tolist(),
                pdms=float(label[-1] * 100),
                planned_space_ok=bool(space(path)),
            )
            keys.append(name + "/" + kind)
            all_labels.append(label)
            all_paths.append(path)
        details[name] = dict(
            route_lane_ids=c.route_lane_ids,
            center_covered=bool(polygon.covers(center)),
            center_distance_m=float(polygon.distance(center)),
            paths=paths,
            official_original_labels=name == "cached",
            diagnostic_only=name != "cached",
        )
    reference_result = None
    if anchor_trace and anchor_trace["connected"]:
        # Rebuild reference and centerline together to test the cache semantics.
        # This is an isolated planner instance, with no module/global patch.
        import os, lzma, pickle
        from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
        from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor

        scenario = NavSimScenario(
            scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
        )
        processor = MetricCacheProcessor(None, False)
        inputs, initialization = processor._get_planner_inputs(scenario)
        planner = processor._pdm_closed
        planner.initialize(initialization)
        planner._route_roadblock_correction = lambda ego: planner._load_route_dicts(
            anchor_trace["after"]
        )
        reference = planner.compute_planner_trajectory(inputs)
        candidate_cache = copy(cache)
        candidate_cache.trajectory = reference
        candidate_cache.centerline = planner._centerline
        candidate_cache.route_lane_ids = list(planner._route_lane_dict)
        candidate_cache.drivable_area_map = planner._drivable_area_map
        cache_path = output / "diagnostic_caches" / (token + ".pkl.xz")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_cache.file_path = cache_path
        atomic(
            cache_path, lambda f: f.write(lzma.compress(pickle.dumps(candidate_cache), preset=0))
        )
        ev = Evaluator(candidate_cache)
        check = planned_space_checker(candidate_cache)
        reference_result = dict(diagnostic_only=True, official_original_labels=False, paths={})
        for kind, path in tests.items():
            label = ev.score(path)
            reference_result["paths"][kind] = dict(
                pdms=float(label[-1] * 100),
                labels=label.tolist(),
                planned_space_ok=bool(check(path)),
            )
            keys.append("rebuilt_anchor_reference/" + kind)
            all_labels.append(label)
            all_paths.append(path)
    assert tuple(cache.route_lane_ids) == original_route
    result = dict(
        token=token,
        record=record,
        nearby_directed_lanes=lanes,
        history_anchors=history,
        anchor=anchor_trace,
        variants=details,
        rebuilt_anchor_reference=reference_result,
        source_modified=False,
        production_route_modified=False,
        acceptance_policy_modified=False,
        diagnostic_labels_not_eligible_for_training=True,
    )
    out = output / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / (token + ".json"), result)
    atomic(
        out / (token + ".npz"),
        lambda f: np.savez(
            f,
            names=np.asarray(keys, dtype="U100"),
            trajectories=np.asarray(all_paths, np.float32),
            labels=np.asarray(all_labels, float),
        ),
    )
    if "directed_anchor_suffix" in details:
        render_comparison(cache, expert, result, output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokens", "trace-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    tokens = a.tokens.read_text().split()
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid tokens")
    a.output.mkdir(parents=True, exist_ok=True)
    config = dict(
        version=3,
        tokens=sorted(tokens),
        source=str(a.source),
        trace_root=str(a.trace_root),
        heading_error_deg=45,
        baseline_distance_m=3.0,
        diagnostic_only=True,
    )
    cfg = a.output / "config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError("Resume differs")
    cfg.write_text(json.dumps(config, indent=2) + "\n")
    records = {
        r["token"]: r for r in map(json.loads, (a.source / "scenes.jsonl").read_text().splitlines())
    }
    results = []
    for t in tokens:
        result = inspect(records[t], a.source, a.trace_root, a.output)
        results.append(result)
        print(
            json.dumps(
                dict(
                    token=t,
                    anchor=result["anchor"],
                    center_covered={k: v["center_covered"] for k, v in result["variants"].items()},
                )
            ),
            flush=True,
        )
    (a.output / "summary.json").write_text(
        json.dumps(dict(scenes=len(results), results=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
