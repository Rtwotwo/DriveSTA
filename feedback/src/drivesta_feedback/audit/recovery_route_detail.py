"""Read-only trace of route correction and independent initial-pose geometry."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np


def render_scene(scene, cache, expert, raw_lanes, result, output):
    """Show source-map membership separately from actual driving direction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L

    pose = cache.ego_state.rear_axle
    origin = np.array([pose.x, pose.y])
    c, s = np.cos(pose.heading), np.sin(pose.heading)
    rotation = np.array([[c, -s], [s, c]])

    def local(xy):
        return (np.asarray(xy) - origin) @ rotation

    fig, ax = plt.subplots(figsize=(7, 6))
    dm = cache.drivable_area_map
    for i in dm.get_indices_of_map_type([L.LANE, L.LANE_CONNECTOR]):
        k = dm.tokens[i]
        if dm[k].distance(cache.ego_state.car_footprint.geometry) > 22:
            continue
        on_route = k in cache.route_lane_ids
        raw_only = k in raw_lanes and not on_route
        color = "#79b9df" if on_route else "#edb467" if raw_only else "#d7dce0"
        polygons = [dm[k]] if dm[k].geom_type == "Polygon" else list(dm[k].geoms)
        for polygon in polygons:
            ax.add_patch(
                Polygon(
                    local(polygon.exterior.coords),
                    facecolor=color,
                    edgecolor="#62717d",
                    lw=0.35,
                    alpha=0.35,
                )
            )
    box = local(cache.ego_state.car_footprint.geometry.exterior.coords)
    ax.plot(box[:, 0], box[:, 1], color="black", lw=1.7)
    center = local([[cache.ego_state.center.x, cache.ego_state.center.y]])[0]
    ax.scatter(*center, c="#c32630", marker="x", s=65, zorder=8)
    ax.annotate("", xy=(5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.plot(
        np.r_[0, expert[:, 0]],
        np.r_[0, expert[:, 1]],
        "--",
        color="#b12d83",
        lw=1.7,
        label="Recorded expert",
    )
    for lane in [x for x in result["nearby_lanes"] if x["covers_center"]]:
        ax.text(
            0.02,
            0.97
            - 0.065
            * sum(
                x["lane_id"] < lane["lane_id"] for x in result["nearby_lanes"] if x["covers_center"]
            ),
            lane["lane_id"] + " | heading gap %.1f deg" % lane["heading_error_deg"],
            transform=ax.transAxes,
            fontsize=8,
            va="top",
        )
    ax.set(
        xlim=(-10, 22),
        ylim=(-16, 16),
        xlabel="Forward x (m)",
        ylabel="Left y (m)",
        title=result["token"]
        + "\nCenter outside cached route: %.3f m"
        % result["initial_geometry"]["center_distance_outside_route_m"],
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(
        0.08,
        0.015,
        "Blue: cached route; orange: raw route only; grey: other lanes. Red x: vehicle center.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(Path(output) / "scenes" / (result["token"] + ".png"), dpi=150)
    plt.close(fig)


def inspect(job):
    record, source, output = job
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    import shapely
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.utils import route_utils as ru
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        state_array_to_coords_array,
    )
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from drivesta_feedback.batch.source import load_scene, Evaluator
    from drivesta_feedback.batch.recovery_rollout_search import Geometry
    from drivesta_feedback.audit.recovery_remaining import initial_geometry
    import os

    token = record["token"]
    out = Path(output) / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    existing = (
        json.loads((out / (token + ".json")).read_text())
        if (out / (token + ".json")).exists()
        else None
    )
    if existing is not None and (out / (token + ".png")).exists():
        return existing
    folder = Path(source) / "scenes" / record["split"] / token
    contract = json.loads((folder / "contract.json").read_text())
    scene, expert, cache, identity = load_scene(record, [token], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source identity changed")
    scenario = NavSimScenario(
        scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
    )
    raw_ids = list(dict.fromkeys(scenario.get_route_roadblock_ids()))
    blocks = {}
    for k in raw_ids:
        obj = scene.map_api.get_map_object(k, L.ROADBLOCK) or scene.map_api.get_map_object(
            k, L.ROADBLOCK_CONNECTOR
        )
        if obj is not None:
            blocks[k] = obj
    raw_lanes = {lane.id: lane for b in blocks.values() for lane in b.interior_edges}
    if existing is not None:
        render_scene(scene, cache, expert, raw_lanes, existing, output)
        return existing
    start, choices = ru.get_current_roadblock_candidates(
        cache.ego_state.rear_axle, scene.map_api, blocks
    )
    trace = {}
    original_remove = ru.remove_route_loops

    def traced_remove(objects, ids):
        trace["before_loop_removal"] = list(ids)
        result = original_remove(objects, ids)
        trace["after_loop_removal"] = list(result[1])
        return result

    ru.remove_route_loops = traced_remove
    try:
        corrected = ru.route_roadblock_correction(cache.ego_state.rear_axle, scene.map_api, blocks)
    finally:
        ru.remove_route_loops = original_remove
    geometry = Geometry(cache)
    coords = state_array_to_coords_array(
        geometry.states(np.zeros((8, 3), np.float32))[None], get_pacifica_parameters()
    )[0, 0]
    center = shapely.Point(coords[-1])
    native_center = cache.ego_state.car_footprint.center
    native_box = cache.ego_state.car_footprint.geometry
    array_box = shapely.Polygon(coords[:4])
    cached_lanes = set(cache.route_lane_ids)
    independent_route = []
    lanes = {}
    dm = cache.drivable_area_map
    polygon_difference = []
    for i in dm.get_indices_of_map_type([L.LANE, L.LANE_CONNECTOR]):
        k = dm.tokens[i]
        lane = scene.map_api.get_map_object(k, L.LANE) or scene.map_api.get_map_object(
            k, L.LANE_CONNECTOR
        )
        if lane is None:
            raise ValueError("Cached lane missing from source map: " + k)
        if k in cached_lanes:
            independent_route.append(lane.polygon)
            polygon_difference.append(lane.polygon.symmetric_difference(dm[k]).area)
        if lane.polygon.distance(center) <= 12:
            line = lane.baseline_path.linestring
            s = line.project(center)
            a, b = line.interpolate(max(0, s - 0.5)), line.interpolate(min(line.length, s + 0.5))
            heading = np.arctan2(b.y - a.y, b.x - a.x)
            error = abs(
                np.arctan2(
                    np.sin(heading - native_center.heading), np.cos(heading - native_center.heading)
                )
            )
            block = lane.get_roadblock_id()
            lanes[k] = dict(
                lane_id=k,
                roadblock_id=block,
                covers_center=bool(lane.polygon.covers(center)),
                distance_m=float(lane.polygon.distance(center)),
                heading_error_deg=float(np.degrees(error)),
                in_raw=k in raw_lanes,
                in_cached=k in cached_lanes,
                in_before_loop_removal=block in trace["before_loop_removal"],
                in_after_loop_removal=block in trace["after_loop_removal"],
            )
    raw_polygon = shapely.union_all([x.polygon for x in raw_lanes.values()])
    independent_polygon = shapely.union_all(independent_route)
    label, tracked = Evaluator(cache).score(expert, details=True)
    current = initial_geometry(cache)
    result = dict(
        token=token,
        record=record,
        initial_geometry=current,
        source_identity_verified=True,
        raw_route_ids=raw_ids,
        starting_block=start.id,
        starting_candidates=[x.id for x in choices],
        correction_trace=trace,
        corrected_ids=corrected,
        nearby_lanes=sorted(lanes.values(), key=lambda x: x["distance_m"]),
        native_center_error_m=float(
            np.linalg.norm(coords[-1] - [native_center.x, native_center.y])
        ),
        native_footprint_symmetric_difference_m2=float(
            native_box.symmetric_difference(array_box).area
        ),
        maximum_cached_lane_polygon_error_m2=float(max(polygon_difference, default=0)),
        independent_source_route_covers_center=bool(independent_polygon.covers(center)),
        raw_route_covers_center=bool(raw_polygon.covers(center)),
        expert_labels=label.tolist(),
        source_modified=False,
        acceptance_policy_modified=False,
    )
    (out / (token + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    render_scene(scene, cache, expert, raw_lanes, result, output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    tokens = a.tokens.read_text().split()
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("Expected unique nonempty tokens")
    a.output.mkdir(parents=True, exist_ok=True)
    records = {
        r["token"]: r for r in map(json.loads, (a.source / "scenes.jsonl").read_text().splitlines())
    }
    config = dict(source=str(a.source), tokens=sorted(tokens), version=1)
    cfg = a.output / "config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError("Resume configuration differs")
    cfg.write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(inspect, [(records[t], str(a.source), str(a.output)) for t in tokens]):
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        token=row["token"],
                        initial_route=row["independent_source_route_covers_center"],
                        raw_route=row["raw_route_covers_center"],
                        center_error=row["native_center_error_m"],
                    )
                ),
                flush=True,
            )
    result = dict(
        scenes=len(rows), records=rows, source_modified=False, acceptance_policy_modified=False
    )
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
