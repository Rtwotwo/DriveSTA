"""Generate chronological GT-relative families and pair-conditioned EPDMS banks."""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
import lzma
import pickle
from pathlib import Path
import time
import traceback

from navsim.v2.pickle_compat import load as load_v2
import numpy as np
import shapely
from .fast import FastArchiveScorer, trajectory_array
from .scoring import (
    R,
    FIELDS,
    SAMPLING,
    OUTPUT_SAMPLING,
    Trajectory,
    transform_trajectory,
    score_from_components,
)
from .sampling import construct_bank, local_reference, family_parameters, unique_ranked
from ..batch.filter_targets import TargetPolicy, tracking_errors, wrap
from ..batch.physical_sampling import physical_mask
from ..batch.store import write_json, checksum
from navsim.v2.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_two_frame_extended_comfort,
)

SCHEMA = "navsim_gt_family_epdms_v1"
BASE = Path(os.environ.get("EPDMS_BASE_ROOT", str(R / "data/epdms_base")))
POLICY = TargetPolicy(
    max_targets=256,
    max_track_queries=3072,
    min_signed_speed_mps=-1e-6,
    max_backward_distance_m=1e-6,
    check_planned_space=True,
)
DTYPE = np.dtype(
    [
        ("family_id", "<i4"),
        ("trajectories", "<f4", (8, 3)),
        ("components_raw", "<f8", (8,)),
        ("components", "<f8", (8,)),
        ("ec", "<f8"),
        ("epdms", "<f8"),
        ("admissible", "?"),
        ("physical", "?"),
        ("tracking", "?"),
        ("space", "?"),
    ]
)


def ec_pair(current, previous, gap):
    if not 0 < gap < 0.55:
        raise ValueError("Invalid official adjacent-frame gap")
    shift = round(gap / 0.1)
    if shift <= 0:
        raise ValueError("Invalid EC overlap")
    a, b = current[:, :-shift], previous[:, shift:]
    if a.shape != b.shape:
        raise ValueError("Paired trajectory shape mismatch")
    return ego_is_two_frame_extended_comfort(a, b, np.arange(a.shape[1]) * 0.1).astype(float)


def planned_space_many(cache, paths):
    """Batch the existing exact corner/route check; no buffers or route extension."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim.v2.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        state_array_to_coords_array,
    )
    from navsim.v2.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex

    m = cache.drivable_area_map
    roads = m.get_indices_of_map_type(
        [
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.CARPARK_AREA,
        ]
    )
    routes = [i for i, token in enumerate(m.tokens) if token in cache.route_lane_ids]
    road = shapely.union_all([m[m.tokens[i]] for i in roads if m[m.tokens[i]].is_valid])
    route = shapely.union_all([m[m.tokens[i]] for i in routes if m[m.tokens[i]].is_valid])
    if road.is_empty or route.is_empty:
        return np.zeros(len(paths), dtype=bool)
    shapely.prepare(road)
    shapely.prepare(route)
    result = []
    for begin in range(0, len(paths), 128):
        states = np.stack(
            [
                trajectory_array(
                    transform_trajectory(
                        Trajectory(p.astype(float), OUTPUT_SAMPLING), cache.ego_state
                    ),
                    SAMPLING,
                    cache.ego_state.time_point,
                )
                for p in paths[begin : begin + 128]
            ]
        )
        coords = state_array_to_coords_array(states, get_pacifica_parameters())
        corners = shapely.points(coords[:, :, :4, :])
        centers = shapely.points(coords[:, :, BBCoordsIndex.CENTER, :])
        result.append(
            shapely.covers(road, corners).all(axis=(1, 2))
            & shapely.covers(route, centers).all(axis=1)
        )
    return np.concatenate(result) if result else np.zeros(0, dtype=bool)


def admission(cache, paths, raw, states, epdms, expert):
    dyn = cache.ego_state.dynamic_car_state
    velocity = [dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y]
    acceleration = [dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y]
    physical, masks = physical_mask(paths, expert, POLICY, velocity, acceleration)
    # Raw, unfiltered components: no collision, drivable, direction, TLC, TTC, LK, HC.
    safety = (raw[:, [0, 1, 2, 3, 5, 6, 7]] == 1.0).all(axis=1)
    pose = cache.ego_state.rear_axle
    errors = np.array(
        [tracking_errors(p, s, [pose.x, pose.y, pose.heading]) for p, s in zip(paths, states)]
    )
    tracked = (
        errors
        <= [
            POLICY.max_tracking_ade_m,
            POLICY.max_tracking_error_m,
            POLICY.max_tracking_heading_error_rad,
        ]
    ).all(axis=1)
    space = np.zeros(len(paths), dtype=bool)
    # Only candidate targets need the additional spatial test; false means not admitted.
    check = np.flatnonzero(physical & safety & tracked & (epdms > 0.95))
    space[check] = planned_space_many(cache, paths[check])
    admitted = physical & safety & tracked & space & (epdms > 0.95)
    funnel = {
        "raw_score_gt95": int((epdms > 0.95).sum()),
        "raw_safety_gt95": int((safety & (epdms > 0.95)).sum()),
        "physical_gt95": int((physical & safety & (epdms > 0.95)).sum()),
        "tracking_gt95": len(check),
        "space_gt95": int(admitted.sum()),
        "physical_all": int(physical.sum()),
        "forward_motion_all": int(masks["forward_motion"].sum()),
    }
    return admitted, physical, tracked, space, errors, funnel


def atomic_npy(path, array):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    tmp.replace(path)


def atomic_npz(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _read_scene(folder, verify=True):
    folder = Path(folder)
    info = json.loads((folder / "summary.json").read_text())
    if info["schema"] != SCHEMA:
        raise ValueError("Use the EPDMS reader, not the v1 seven-label reader")
    full = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
    with np.load(folder / "high_score.npz", allow_pickle=False) as archive:
        high = {k: archive[k] for k in archive.files}
    if full.dtype != DTYPE or full.shape != (3072,):
        raise ValueError("Invalid full bank")
    ids = high["query_indices"]
    if ids.ndim != 1 or len(ids) != info["selected_count"] or len(ids) > 256:
        raise ValueError("Invalid target count")
    if verify:
        for name, expected in info["files"].items():
            if checksum(folder / name) != expected:
                raise ValueError("Changed scene file: " + name)
        want = unique_ranked(full["trajectories"], full["epdms"], full["admissible"])
        if not np.array_equal(ids, want):
            raise ValueError("Invalid score ordering, cutoff or deduplication")
        for field in ("trajectories", "epdms", "components_raw", "components", "ec", "family_id"):
            if not np.array_equal(full[field][ids], high[field], equal_nan=True):
                raise ValueError("High bank/source mismatch: " + field)
        expected_previous = ids if info["previous_token"] is not None else np.full(len(ids), -1)
        if not np.array_equal(high["previous_family_id"], expected_previous):
            raise ValueError("Previous family slot changed")
        if not np.array_equal(full["family_id"], np.arange(3072)):
            raise ValueError("Family identity reordered")
    return info, full, high


def _consistent_scene_read(folder, reader):
    folder = Path(folder)
    for attempt in range(31):
        before = None
        try:
            before = checksum(folder / "summary.json")
            result = reader()
            if checksum(folder / "summary.json") == before:
                return result
        except FileNotFoundError:
            if attempt == 30:
                raise
        except ValueError:
            try:
                changed = before is not None and checksum(folder / "summary.json") != before
            except FileNotFoundError:
                changed = True
            if not changed:
                raise
        if attempt == 30:
            raise RuntimeError("Scene publication is still in progress; retry after recovery")
        time.sleep(0.1)


def read_scene(folder, verify=True):
    return _consistent_scene_read(folder, lambda: _read_scene(folder, verify))


def read_target_pairs(root, split, token):
    folder = Path(root) / "scenes" / split / token
    return _consistent_scene_read(folder, lambda: _read_target_pairs(root, split, token))


def _read_target_pairs(root, split, token):
    """Load full temporal context; never treat EC as a standalone trajectory label."""
    root = Path(root)
    info, _, high = read_scene(root / "scenes" / split / token)

    def pose_for(record):
        contract = json.loads(
            (root / "scenes" / record["split"] / record["token"] / "contract.json").read_text()
        )
        if "ego_pose_global" in contract:
            return np.asarray(contract["ego_pose_global"], float)
        # Earlier diagnostic artifacts retain an exact official-cache reference.
        with lzma.open(contract["official_cache_path"], "rb") as f:
            cache = pickle.load(f)
        p = cache.ego_state.rear_axle
        return np.array([p.x, p.y, p.heading])

    result = dict(
        current=high,
        previous_trajectories=None,
        previous_ego_pose_global=None,
        current_ego_pose_global=pose_for(info),
        gap_seconds=None,
        previous_coordinate_frame="previous ego rear axle",
        previous_token=info["previous_token"],
    )
    if info["previous_token"] is None:
        return result
    contract = json.loads((root / "scenes" / split / token / "contract.json").read_text())
    context = contract.get("previous_context")
    previous_folder = (
        root / context if context else root / "scenes" / split / info["previous_token"]
    )
    if root.resolve() not in previous_folder.resolve().parents:
        raise ValueError("Previous context must stay inside the corpus")
    prev_info, previous, _ = read_scene(previous_folder)
    if prev_info["token"] != info["previous_token"]:
        raise ValueError("Previous context token mismatch")
    if (
        context
        and checksum(previous_folder / "candidates.npy") != contract["previous_context_sha256"]
    ):
        raise ValueError("Previous context changed")
    previous_contract = json.loads((previous_folder / "contract.json").read_text())
    result.update(
        previous_trajectories=np.array(previous["trajectories"][high["previous_family_id"]]),
        previous_ego_pose_global=np.asarray(previous_contract["ego_pose_global"]),
        gap_seconds=info["start_time"] - prev_info["start_time"],
    )
    return result


def cache_for(row):
    path = BASE / "metric_cache" / row["log_name"] / "unknown" / row["token"] / "metric_cache.pkl"
    with lzma.open(path, "rb") as f:
        cache = load_v2(f)
    return cache, path


def process_log(task):
    rows, out_string, seed, contract = task
    out = Path(out_string)
    parameters = family_parameters(seed)
    previous, previous_time, previous_token = None, None, None
    summaries = []
    # Validate a contiguous committed prefix, then only re-simulate its last frame.
    for row in rows:
        folder = out / "scenes" / row["split"] / row["token"]
        if not (folder / "summary.json").exists():
            break
        info, _, _ = read_scene(folder)
        prev = summaries[-1] if summaries else None
        predecessor = (
            prev["token"] if prev and 0 < info["start_time"] - prev["start_time"] < 0.55 else None
        )
        if info["contract"] != contract or info["previous_token"] != predecessor:
            raise ValueError("Changed run or adjacency on resume")
        summaries.append(info)
    if len(summaries) == len(rows):
        return summaries
    if summaries:
        last = summaries[-1]
        cache, _ = cache_for(last)
        _, full, _ = read_scene(out / "scenes" / last["split"] / last["token"])
        previous = FastArchiveScorer(cache).simulate(full["trajectories"])
        previous_time, previous_token = last["start_time"], last["token"]
    for row in rows[len(summaries) :]:
        folder = out / "scenes" / row["split"] / row["token"]
        cache, cache_path = cache_for(row)
        now = cache.timepoint.time_s
        adjacent = previous_time is not None and 0 < now - previous_time < 0.55
        predecessor = previous_token if adjacent else None
        start = time.time()
        engine = FastArchiveScorer(cache)
        expert = np.asarray(cache.human_trajectory.poses)
        pose = cache.ego_state.rear_axle
        pdm_reference = local_reference(engine.ref[5::5, :3], [pose.x, pose.y, pose.heading])
        paths = construct_bank(expert, pdm_reference, parameters)
        raw, components, states = engine.score(paths, chunk=128, return_states=True)
        ec = ec_pair(states, previous, now - previous_time) if adjacent else np.full(3072, np.nan)
        epdms = (
            score_from_components(components, ec) if adjacent else score_from_components(components)
        )
        valid, physical, tracked, space, errors, funnel = admission(
            cache, paths, raw, states, epdms, expert
        )
        full = np.zeros(3072, dtype=DTYPE)
        full["family_id"] = np.arange(3072)
        for field, value in [
            ("trajectories", paths),
            ("components_raw", raw),
            ("components", components),
            ("ec", ec),
            ("epdms", epdms),
            ("admissible", valid),
            ("physical", physical),
            ("tracking", tracked),
            ("space", space),
        ]:
            full[field] = value
        ids = unique_ranked(paths, epdms, valid)
        selected_xy = paths[ids, :, :2]
        if len(ids) > 1:
            distances = np.linalg.norm(selected_xy[:, None] - selected_xy[None, :], axis=-1).mean(
                -1
            )
            np.fill_diagonal(distances, np.inf)
            nearest_ade = float(np.median(distances.min(axis=1)))
        else:
            nearest_ade = None
        folder.mkdir(parents=True, exist_ok=True)
        atomic_npy(folder / "candidates.npy", full)
        atomic_npz(
            folder / "high_score.npz",
            query_indices=ids,
            **{
                k: full[k][ids]
                for k in (
                    "trajectories",
                    "components_raw",
                    "components",
                    "ec",
                    "epdms",
                    "family_id",
                )
            },
            previous_family_id=ids if adjacent else np.full(len(ids), -1),
            tracking_errors=errors[ids],
        )
        write_json(
            folder / "contract.json",
            dict(
                schema=SCHEMA,
                scoring="corrected NAVTEST single-stage non-reactive",
                official_cache_path=str(cache_path),
                threshold="epdms > 0.95",
                previous_token=predecessor,
                ego_pose_global=[pose.x, pose.y, pose.heading],
                start_time=now,
                pairing="same family_id, before score selection",
                fields=FIELDS,
                policy=asdict(POLICY),
                ec_is_context_dependent=True,
                units="raw 0..1",
                seed=seed,
                run_contract=contract,
            ),
        )
        write_json(
            folder / "expert.json",
            dict(
                trajectory=expert.tolist(),
                components=engine.expert_components.tolist(),
                pdm_reference=pdm_reference.tolist(),
            ),
        )
        old = json.loads(
            (BASE / "scenes" / row["split"] / row["token"] / "summary.json").read_text()
        )
        info = dict(
            schema=SCHEMA,
            contract=contract,
            token=row["token"],
            split=row["split"],
            log_name=row["log_name"],
            start_time=now,
            previous_token=predecessor,
            ec_defined=adjacent,
            candidate_count=3072,
            selected_count=len(ids),
            min_selected_epdms=float(epdms[ids].min() * 100) if len(ids) else None,
            ec_pass_count=int(ec.sum()) if adjacent else None,
            best_epdms=float(epdms.max() * 100),
            old_ge95=old["ge95_count"],
            old_selected_without_new_safety=old["selected_count"],
            old_ec_pass=old["ec_pass_count"],
            funnel=funnel,
            elapsed_s=time.time() - start,
            median_nearest_target_ade_m=nearest_ade,
            files={
                name: checksum(folder / name)
                for name in ["candidates.npy", "high_score.npz", "contract.json", "expert.json"]
            },
        )
        write_json(folder / "summary.json", info)  # Publication/commit marker, written last.
        read_scene(folder)
        summaries.append(info)
        print(
            json.dumps(
                {
                    k: info[k]
                    for k in ["token", "selected_count", "ec_pass_count", "funnel", "elapsed_s"]
                }
            ),
            flush=True,
        )
        previous, previous_time, previous_token = states, now, row["token"]
    write_json(
        out / "logs" / (rows[0]["log_name"] + ".json"), dict(complete=True, summaries=summaries)
    )
    return summaries


def build_cohort(log_count, limit):
    # One compact sequential read avoids 1192 small-file NAS round trips.
    mapping = {}
    with gzip.open(BASE / "scene_summary.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            row["start_time"] = float(row["start_time"])
            mapping.setdefault(row["log_name"], []).append(row)
    logs = sorted(mapping)
    if sum(map(len, mapping.values())) != 103288:
        raise ValueError("Source census is not the full103288 population")
    if log_count:
        candidates = [key for key in logs if len(mapping[key]) >= limit]
        ids = np.random.default_rng(20261028).choice(len(candidates), log_count, replace=False)
        logs = [candidates[i] for i in sorted(ids)]
    groups = []
    for name in logs:
        rows = mapping[name]
        rows.sort(key=lambda r: r["start_time"])
        if limit:
            rows = rows[:limit]
        groups.append(rows)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--logs", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    if args.workers < 1 or args.logs < 0 or args.frames < 0 or bool(args.logs) != bool(args.frames):
        raise ValueError("Use both --logs and --frames for a pilot")
    os.nice(10)
    args.out.mkdir(parents=True, exist_ok=True)
    handle = (args.out / ".writer.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    groups = build_cohort(args.logs, args.frames)
    config = dict(
        schema=SCHEMA,
        seed=args.seed,
        policy=asdict(POLICY),
        construction="GT 48x48 and PDM-reference32x24 grids, both axes0.05m; bounded interior perturbation",
        groups=[[r["token"] for r in rows] for rows in groups],
        strict_threshold=0.95,
        files={p.name: checksum(p) for p in Path(__file__).parent.glob("*.py")},
    )
    contract = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = args.out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Output has different code or population; keep prior outputs")
    write_json(path, config)
    atomic_npz(
        args.out / "families.npz",
        coefficients=family_parameters(args.seed),
        family_id=np.arange(3072),
    )
    start = time.time()
    results = []
    failures = []
    expected = sum(len(rows) for rows in groups)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_log, (rows, str(args.out), args.seed, contract)): rows[0][
                "log_name"
            ]
            for rows in groups
        }
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception:
                failures.append(dict(log_name=futures[future], traceback=traceback.format_exc()))
                print(json.dumps(failures[-1]), flush=True)
            write_json(
                args.out / "status.json",
                dict(
                    processed=len(results),
                    expected=expected,
                    high256=sum(s["selected_count"] == 256 for s in results),
                    shortfall_processed=sum(s["selected_count"] < 256 for s in results),
                    targets=sum(s["selected_count"] for s in results),
                    trajectory_records=len(results) * 3072,
                    completion_unit="completed logs; active scene files are not counted",
                    errors=failures,
                    elapsed_s=time.time() - start,
                    updated_unix=time.time(),
                ),
            )
            shortfalls = args.out / "shortfall_tokens.txt.tmp"
            shortfalls.write_text(
                "".join(
                    s["token"] + "\n"
                    for s in sorted(results, key=lambda s: s["token"])
                    if s["selected_count"] < 256
                )
            )
            shortfalls.replace(args.out / "shortfall_tokens.txt")
    summary = dict(
        complete=not failures and len(results) == expected,
        processed=len(results),
        expected=expected,
        high256=sum(s["selected_count"] == 256 for s in results),
        with_targets=sum(s["selected_count"] > 0 for s in results),
        targets=sum(s["selected_count"] for s in results),
        ec_pass=sum(s["ec_pass_count"] or 0 for s in results),
        ec_count=sum(s["ec_defined"] for s in results) * 3072,
        old_high256=sum(s["old_selected_without_new_safety"] == 256 for s in results),
        old_ec_pass=sum((s["old_ec_pass"] or 0) for s in results if s["ec_defined"]),
        errors=failures,
        elapsed_s=time.time() - start,
        scenes=results,
    )
    write_json(args.out / "summary.json", summary)
    (args.out / "shortfall_tokens.txt").write_text(
        "".join(
            s["token"] + "\n"
            for s in sorted(results, key=lambda s: s["token"])
            if s["selected_count"] < 256
        )
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "scenes"}), flush=True)
    if failures:
        raise RuntimeError("Some logs failed; partial files retained")


if __name__ == "__main__":
    main()
