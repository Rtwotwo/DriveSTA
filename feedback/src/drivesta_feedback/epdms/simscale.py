"""SimScale EPDMS construction with a separately recorded inferred-tail gate.

Synthetic clips have one eligible anchor each: EC is unavailable, never one.
Official within-window EPDMS is unchanged. Full-horizon TTC is an extra target
qualification test against both previously validated 4--5s actor forecasts.
"""

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
from dataclasses import asdict
import fcntl
import hashlib
from navsim.common.release_paths import metadata_json as json
import lzma
import pickle
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
from .scoring import R, SAMPLING, OUTPUT_SAMPLING, Trajectory, score_from_components
from .fast import FastArchiveScorer
from .dataset import (
    DTYPE,
    SCHEMA,
    POLICY,
    admission,
    atomic_npy,
    atomic_npz,
    read_scene,
    write_json,
    checksum,
)
from .sampling import family_parameters, construct_bank, local_reference, unique_ranked
from .tail_gate import FullHorizonTTC

SOURCE = Path(os.environ.get("EPDMS_SIMSCALE_SOURCE", str(R / "data/trajectories/simscale")))
EXPECTED = 204831
PROTOCOL = "simscale_official_epdms_unpaired_with_inferred_tail_gate_v1"


def representative_indices(paths, scores, eligible, capacity=32, min_ade=0.25):
    """Full-trajectory representatives, matching V2's 0.25m ADE separation."""
    ids = np.flatnonzero(np.asarray(eligible) & (np.asarray(scores) > 0.95))
    if not len(ids):
        return np.zeros(0, dtype=np.int64)
    xy = np.asarray(paths, dtype=np.float64)[ids, :, :2]
    first = np.lexsort((ids, -np.asarray(scores)[ids]))[0]
    chosen = [int(first)]
    distance = np.linalg.norm(xy - xy[first], axis=-1).mean(-1)
    while len(chosen) < capacity and distance.max() >= min_ade:
        next_id = int(np.argmax(distance))
        chosen.append(next_id)
        distance = np.minimum(distance, np.linalg.norm(xy - xy[next_id], axis=-1).mean(-1))
    return ids[chosen]


def training_targets(folder, epoch=0, count=8):
    """Masked generation targets only; EPDMS is not a legacy seven-label PDMS tensor."""
    if count < 1 or epoch < 0:
        raise ValueError("Invalid target view")
    info, full, _ = read_simscale_scene(folder)
    with np.load(Path(folder) / "generation_targets.npz", allow_pickle=False) as archive:
        bank = archive["query_indices"]
    n = len(bank)
    take = min(n, count)
    indices = (np.arange(take) + epoch * count) % n if n else np.zeros(0, dtype=np.int64)
    trajectories = np.zeros((count, 8, 3), dtype=np.float32)
    mask = np.zeros(count, dtype=bool)
    ids = np.full(count, -1, dtype=np.int64)
    ids[:take] = bank[indices]
    trajectories[:take] = full["trajectories"][ids[:take]]
    mask[:take] = True
    return dict(
        trajectories=trajectories,
        mask=mask,
        query_indices=ids,
        token=info["token"],
        protocol=PROTOCOL,
        available_targets=n,
    )


def export_views(out, results):
    """Publish explicit eight-target and full-256 views without discarding partial banks."""
    out = Path(out)
    ordered = sorted(results, key=lambda r: r["token"])
    for name, predicate in [
        ("shortfall_tokens.txt", lambda n: n < 256),
        ("under8_tokens.txt", lambda n: n < 8),
        ("train_at_least8_tokens.txt", lambda n: n >= 8),
        ("train_full256_tokens.txt", lambda n: n == 256),
    ]:
        tmp = out / (name + ".tmp")
        tmp.write_text(
            "".join(r["token"] + "\n" for r in ordered if predicate(r["selected_count"]))
        )
        tmp.replace(out / name)
    for name, predicate in [
        ("train_diverse8_tokens.txt", lambda n: n >= 8),
        ("under_diverse8_tokens.txt", lambda n: n < 8),
    ]:
        tmp = out / (name + ".tmp")
        tmp.write_text(
            "".join(r["token"] + "\n" for r in ordered if predicate(r["representative_count"]))
        )
        tmp.replace(out / name)
    names = [
        "token",
        "split",
        "source_group",
        "selected_count",
        "representative_count",
        "missing_to8",
        "missing_to256",
        "best_epdms",
        "raw_score_gt95",
        "raw_safety_gt95",
        "physical_gt95",
        "tracking_gt95",
        "space_gt95",
        "both_tail_gt95",
    ]
    tmp = out / "shortfall_details.csv.tmp"
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for r in ordered:
            if r["selected_count"] == 256:
                continue
            line = {k: r[k] for k in names if k in r}
            line.update({k: r["funnel"].get(k) for k in names if k.endswith("_gt95")})
            line.update(
                missing_to8=max(0, 8 - r["selected_count"]), missing_to256=256 - r["selected_count"]
            )
            writer.writerow(line)
    tmp.replace(out / "shortfall_details.csv")
    tmp = out / "training_manifest.jsonl.tmp"
    with tmp.open("w") as stream:
        for r in ordered:
            if r["representative_count"] < 8:
                continue
            folder = out / "scenes" / r["split"] / r["token"]
            stream.write(
                json.dumps(
                    dict(
                        token=r["token"],
                        split=r["split"],
                        source_group=r["source_group"],
                        scene_path=str(folder),
                        high_score_path=str(folder / "high_score.npz"),
                        candidates_path=str(folder / "candidates.npy"),
                        contract_path=str(folder / "contract.json"),
                        generation_targets_path=str(folder / "generation_targets.npz"),
                        selected_count=r["selected_count"],
                        representative_count=r["representative_count"],
                        protocol=PROTOCOL,
                        label_fields="eight EPDMS components plus separate EC and tail gates",
                    )
                )
                + "\n"
            )
    tmp.replace(out / "training_manifest.jsonl")


class LegacyCacheUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "navsim.planning.metric_caching.train_metric_chache" and name == "MetricCache":
            return SimpleNamespace
        if module.startswith("navsim.") and not module.startswith("navsim.v2."):
            module = "navsim.v2." + module[len("navsim.") :]
        return super().find_class(module, name)


def load_legacy(path):
    # Legacy numpy2 archives use this alias; do not install another environment.
    import sys, numpy.core.numeric

    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    with lzma.open(path, "rb") as f:
        return LegacyCacheUnpickler(f).load()


def source_contract(row):
    path = SOURCE / "scenes" / row["split"] / row["token"] / "contract.json"
    identity = json.loads(path.read_text())["identity"]
    for key in ("token", "log_name", "source", "log_path"):
        if identity[key] != row[key]:
            raise ValueError("Source identity mismatch: " + key)
    return path, identity


def detections_from_occupancy(observation):
    """Preserve the saved measured polygons and object identities at 0--4s."""
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

    tracks = []
    lights = []
    for i in range(41):
        occupancy = observation[i]
        objects = []
        tl_tokens = []
        tl_polys = []
        for token in occupancy.tokens:
            poly = occupancy[token]
            if observation.red_light_token in token:
                tl_tokens.append(token)
                tl_polys.append(poly)
                continue
            original = observation.unique_objects[token]
            coords = np.asarray(poly.exterior.coords)
            base = np.asarray(original.box.geometry.exterior.coords)
            edge = coords[1] - coords[0]
            ref = base[1] - base[0]
            yaw = (
                original.center.heading + np.arctan2(edge[1], edge[0]) - np.arctan2(ref[1], ref[0])
            )
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
            center = poly.centroid
            obj = copy.copy(original)
            obj._box = OrientedBox(
                StateSE2(center.x, center.y, float(yaw)),
                original.box.length,
                original.box.width,
                original.box.height,
            )
            if obj.box.geometry.hausdorff_distance(poly) > 1e-6:
                raise ValueError("Measured actor polygon reconstruction changed")
            objects.append(obj)
        tracks.append(DetectionsTracks(TrackedObjects(objects)))
        lights.append((tl_tokens, np.asarray(tl_polys, dtype=object)))
    return tracks, lights


def build_cache(row, output):
    from navsim.v2.common.dataclasses import Scene, SensorConfig
    from navsim.v2.common.enums import SceneFrameType
    from navsim.v2.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.v2.planning.metric_caching.metric_cache import MapParameters
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

    contract_path, identity = source_contract(row)
    path = Path(identity["official_cache_path"])
    old = load_legacy(path)
    meta = old._actor_tail_meta
    if (
        meta["observed_end_s"] != 4.0
        or meta["inferred_end_s"] != 5.0
        or not meta["prefix_preserved"]
    ):
        raise ValueError("Invalid actor-tail provenance")
    for obs in (old.observation, old._alternative_observation):
        if len(obs._global_to_local_idcs) < 51:
            raise ValueError("Missing one-second actor tail")
    with open(row["log_path"], "rb") as f:
        frames = pickle.load(f)
    ids = [i for i, f in enumerate(frames) if f["token"] == row["token"]]
    if len(ids) != 1 or ids[0] < 3 or ids[0] + 8 >= len(frames):
        raise ValueError("Incomplete 4+8 source frames")
    window = copy.deepcopy(frames[ids[0] - 3 : ids[0] + 9])
    # SimScale has no lidar files. No sensor is loaded by this metric adapter.
    for frame in window:
        if frame["lidar_path"] is None:
            frame["lidar_path"] = ""
    scene = Scene.from_scene_dict_list(window, None, 4, 8, SensorConfig.build_no_sensors())
    scenario = NavSimScenario(
        scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
    )
    origin = old.ego_state.rear_axle
    p = np.asarray([origin.x, origin.y, origin.heading])
    if np.max(np.abs(p - np.asarray(identity["ego_pose"]))) > 1e-4:
        raise ValueError("Cache/source pose mismatch")
    if scenario.start_time.time_us != identity["timestamp_us"]:
        raise ValueError("Source timestamp mismatch")
    expected = scene.get_future_trajectory(8).poses
    expert = np.asarray(identity["expert_gt"], dtype=np.float32)
    if not np.allclose(expected, expert, atol=1e-4, rtol=0):
        raise ValueError("Pseudo-expert mismatch")
    tracks, lights = detections_from_occupancy(old.observation)
    cache = copy.copy(old)
    if not hasattr(cache.centerline, "linestring"):
        from navsim.v2.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
        from nuplan.common.actor_state.state_representation import StateSE2

        coords = np.asarray(cache.centerline.coords)
        delta = np.gradient(coords, axis=0)
        heading = np.arctan2(delta[:, 1], delta[:, 0])
        line = cache.centerline
        cache.centerline = PDMPath(
            [StateSE2(float(x), float(y), float(h)) for (x, y), h in zip(coords, heading)]
        )
        if not cache.centerline.linestring.equals_exact(line, 1e-9):
            raise ValueError("Centerline geometry changed during adaptation")
    cache.observation = copy.copy(old.observation)
    cache.observation._observation_samples = 40
    cache.observation._global_to_local_idcs = list(range(41))
    cache.observation._occupancy_maps = list(old.observation._occupancy_maps[:41])
    cache.observation._detections_tracks = tracks
    cache.observation._occupancy_maps_tl = lights
    cache.log_name = row["log_name"]
    cache.timepoint = old.ego_state.time_point
    cache.scene_type = SceneFrameType.SYNTHETIC
    cache.human_trajectory = Trajectory(expert.astype(float), OUTPUT_SAMPLING)
    cache.past_human_trajectory = InterpolatedTrajectory(
        list(scenario.get_ego_past_trajectory(0, 1.5))
    )
    cache.map_parameters = MapParameters(
        os.environ["NUPLAN_MAPS_ROOT"], "nuplan-maps-v1.0", scenario.map_api.map_name
    )
    cache.current_tracked_objects = tracks[:1]
    cache.future_tracked_objects = tracks[1:]
    cache.past_detections_tracks = list(scenario.get_past_tracked_objects(0, 1.5, num_samples=3))[
        :-1
    ]
    cache._nominal_tail_observation = copy.copy(old.observation)
    cache._alternative_tail_observation = copy.copy(old._alternative_observation)
    cache.file_path = Path(output) / "metric_cache" / row["token"] / "metric_cache.pkl"
    provenance = dict(
        source_contract=str(contract_path),
        source_contract_sha256=checksum(contract_path),
        source_cache=str(path),
        source_cache_sha256=checksum(path),
        source_log=row["log_path"],
        source_log_sha256=checksum(Path(row["log_path"])),
        actor_tail_policy=meta["policy"],
        label_environment="observed0-4s; inferred4-5s used only by extra TTC qualification",
        scene_type="SYNTHETIC",
        human_penalty_filter=False,
        ec_available=False,
        ec_reason="one independent training anchor per synthetic log; no valid adjacent pair",
        actor_tail_actors=meta["actor_count"],
    )
    return cache, provenance


class SyntheticScorer(FastArchiveScorer):
    def __init__(self, cache):
        from .scoring import (
            PDMSimulator,
            PDMScorer,
            PDMScorerConfig,
            LogReplayTrafficAgents,
            get_trajectory_as_array,
            pdm_score,
            component_rows,
        )
        from .fast import ArrayPairScorer
        import shapely

        shapely.prepare(cache.drivable_area_map._geometries)
        self.cache = cache
        self.sim = PDMSimulator(SAMPLING)
        self.scorer = ArrayPairScorer(SAMPLING, PDMScorerConfig(human_penalty_filter=False))
        self.official = PDMScorer(SAMPLING, PDMScorerConfig(human_penalty_filter=False))
        self.traffic = LogReplayTrafficAgents(SAMPLING)
        self.ref = get_trajectory_as_array(cache.trajectory, SAMPLING, cache.ego_state.time_point)
        self.expert_row, self.expert_states = pdm_score(
            cache, cache.human_trajectory, SAMPLING, self.sim, self.official, self.traffic
        )
        self.environment = self.traffic.simulate_environment(self.expert_states, cache)
        self.expert_components = component_rows([self.expert_row])[0]
        self.filter_mask = np.zeros(8, dtype=bool)
        coordinates = np.asarray(cache.centerline.linestring.coords)
        self.scorer._centerline_segments = shapely.STRtree(
            shapely.linestrings(np.stack([coordinates[:-1], coordinates[1:]], axis=1))
        )


def tail_qualification(cache, states):
    result = []
    for observation in (cache._nominal_tail_observation, cache._alternative_tail_observation):
        from .scoring import PDMScorerConfig

        scorer = FullHorizonTTC(SAMPLING, PDMScorerConfig(human_penalty_filter=False))
        scorer._reset(
            states,
            observation,
            cache.centerline,
            cache.route_lane_ids,
            cache.drivable_area_map,
            None,
        )
        scorer._calculate_ego_area()
        scorer._calculate_ttc()
        from navsim.v2.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            WeightedMetricIndex,
        )

        result.append(scorer._weighted_metrics[WeightedMetricIndex.TTC].copy())
    return np.stack(result, axis=-1)


def read_simscale_scene(folder, verify=True):
    """Read EPDMS and its separate inferred-tail qualification without relabeling."""
    folder = Path(folder)
    info, full, high = read_scene(folder, verify=verify)
    if info.get("protocol") != PROTOCOL or info["ec_defined"] or info["previous_token"] is not None:
        raise ValueError("Wrong synthetic scoring protocol or invented EC pairing")
    if not np.isnan(full["ec"]).all() or not np.isnan(high["ec"]).all():
        raise ValueError("Independent synthetic clips must mark EC unavailable")
    if not np.array_equal(score_from_components(full["components"]), full["epdms"]):
        raise ValueError("Saved total does not use official missing-EC normalization")
    if not np.array_equal(full["components_raw"], full["components"]):
        raise ValueError("Synthetic labels must not apply human failure exemptions")
    if not np.array_equal(full["components_raw"][high["query_indices"]], high["components_raw"]):
        raise ValueError("High-bank raw components disagree with full bank")
    with np.load(folder / "tail_checks.npz", allow_pickle=False) as tail:
        pair = np.stack([tail["ttc_nominal"], tail["ttc_alternative"]], axis=-1)
        if pair.shape != (3072, 2) or not np.array_equal(
            tail["evaluated"], np.isfinite(pair).all(axis=1)
        ):
            raise ValueError("Invalid inferred-tail availability mask")
        selected = pair[high["query_indices"]]
        if not np.all(selected == 1.0) or not np.all(full["epdms"][high["query_indices"]] > 0.95):
            raise ValueError("High bank violates strict score or either tail gate")
        if not np.array_equal(selected[:, 0], high["tail_ttc_nominal"]) or not np.array_equal(
            selected[:, 1], high["tail_ttc_alternative"]
        ):
            raise ValueError("High bank tail labels disagree with full bank")
    if "representative_count" in info:
        with np.load(folder / "generation_targets.npz", allow_pickle=False) as archive:
            ids = archive["query_indices"]
        if (
            len(ids) != info["representative_count"]
            or len(ids) > 32
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Invalid representative count")
        if not full["admissible"][ids].all() or not (pair[ids] == 1).all():
            raise ValueError("Invalid generation target admission")
        if len(ids) > 1:
            xy = full["trajectories"][ids, :, :2].astype(float)
            d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1).mean(-1)
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.25:
                raise ValueError("Insufficient representative ADE separation")
    return info, full, high


def process_scene(task):
    row, out_string, seed, run_hash = task
    out = Path(out_string)
    folder = out / "scenes" / row["split"] / row["token"]
    if (folder / "summary.json").exists():
        info, _, _ = read_simscale_scene(folder)
        if info["contract"] != run_hash:
            raise ValueError("Changed resume contract")
        saved = json.loads((folder / "contract.json").read_text())
        if checksum(Path(saved["official_cache_path"])) != saved["official_cache_sha256"]:
            raise ValueError("Changed adapted metric cache")
        if checksum(Path(saved["source_contract"])) != saved["source_contract_sha256"]:
            raise ValueError("Source scene contract changed")
        return info
    started = time.time()
    cache, provenance = build_cache(row, out)
    engine = SyntheticScorer(cache)
    expert = cache.human_trajectory.poses
    pose = cache.ego_state.rear_axle
    reference = local_reference(engine.ref[5::5, :3], [pose.x, pose.y, pose.heading])
    paths = construct_bank(expert, reference, family_parameters(seed))
    raw, components, states = engine.score(paths, return_states=True)
    epdms = score_from_components(components)
    valid, physical, tracked, space, errors, funnel = admission(
        cache, paths, raw, states, epdms, expert
    )
    ids = np.flatnonzero(valid)
    tail = np.full((3072, 2), np.nan)
    for begin in range(0, len(ids), 128):
        selected = ids[begin : begin + 128]
        tail[selected] = tail_qualification(cache, states[selected])
    valid &= (tail == 1).all(axis=1)
    funnel["both_tail_gt95"] = int(valid.sum())
    full = np.zeros(3072, dtype=DTYPE)
    full["family_id"] = np.arange(3072)
    for field, value in [
        ("trajectories", paths),
        ("components_raw", raw),
        ("components", components),
        ("ec", np.full(3072, np.nan)),
        ("epdms", epdms),
        ("admissible", valid),
        ("physical", physical),
        ("tracking", tracked),
        ("space", space),
    ]:
        full[field] = value
    selected = unique_ranked(paths, epdms, valid)
    representatives = representative_indices(paths, epdms, valid)
    folder.mkdir(parents=True, exist_ok=True)
    # Save adapted cache before publication: later scoring/reading has full provenance.
    cache.file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.file_path.with_suffix(".tmp")
    tmp.write_bytes(lzma.compress(pickle.dumps(cache, pickle.HIGHEST_PROTOCOL), preset=0))
    tmp.replace(cache.file_path)
    atomic_npy(folder / "candidates.npy", full)
    atomic_npz(
        folder / "tail_checks.npz",
        ttc_nominal=tail[:, 0],
        ttc_alternative=tail[:, 1],
        evaluated=np.isfinite(tail).all(axis=1),
    )
    atomic_npz(
        folder / "high_score.npz",
        query_indices=selected,
        **{
            k: full[k][selected]
            for k in ("trajectories", "components_raw", "components", "ec", "epdms", "family_id")
        },
        previous_family_id=np.full(len(selected), -1),
        tracking_errors=errors[selected],
        tail_ttc_nominal=tail[selected, 0],
        tail_ttc_alternative=tail[selected, 1],
    )
    atomic_npz(folder / "generation_targets.npz", query_indices=representatives)
    write_json(
        folder / "contract.json",
        dict(
            schema=SCHEMA,
            protocol=PROTOCOL,
            run_contract=run_hash,
            official_cache_path=str(cache.file_path),
            official_cache_sha256=checksum(cache.file_path),
            threshold="epdms > 0.95",
            previous_token=None,
            ego_pose_global=[pose.x, pose.y, pose.heading],
            start_time=cache.timepoint.time_s,
            fields=__import__(__package__ + ".scoring", fromlist=["FIELDS"]).FIELDS,
            policy=asdict(POLICY),
            ec_is_context_dependent=True,
            units="raw 0..1",
            seed=seed,
            **provenance,
        ),
    )
    write_json(
        folder / "expert.json",
        dict(
            trajectory=expert.tolist(),
            components=engine.expert_components.tolist(),
            pdm_reference=reference.tolist(),
        ),
    )
    info = dict(
        schema=SCHEMA,
        protocol=PROTOCOL,
        contract=run_hash,
        token=row["token"],
        split=row["split"],
        log_name=row["log_name"],
        source_group=row["source_group"],
        start_time=cache.timepoint.time_s,
        previous_token=None,
        ec_defined=False,
        candidate_count=3072,
        selected_count=len(selected),
        representative_count=len(representatives),
        representative_min_ade_m=0.25,
        min_selected_epdms=float(epdms[selected].min() * 100) if len(selected) else None,
        best_epdms=float(epdms.max() * 100),
        ec_pass_count=None,
        funnel=funnel,
        elapsed_s=time.time() - started,
        files={
            name: checksum(folder / name)
            for name in [
                "candidates.npy",
                "high_score.npz",
                "generation_targets.npz",
                "tail_checks.npz",
                "contract.json",
                "expert.json",
            ]
        },
    )
    write_json(folder / "summary.json", info)
    read_simscale_scene(folder)
    return info


def read_manifest():
    rows = [json.loads(l) for l in (SOURCE / "scenes.jsonl").read_text().splitlines()]
    if len(rows) != EXPECTED or len({r["token"] for r in rows}) != EXPECTED:
        raise ValueError("Invalid SimScale population")
    if len({r["log_name"] for r in rows}) != EXPECTED:
        raise ValueError("Revisit EC policy: repeated synthetic logs")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--pilot-per-source", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261028)
    a = p.parse_args()
    if a.workers < 1 or a.pilot_per_source < 0:
        raise ValueError("Invalid budget")
    os.nice(10)
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out / ".writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = read_manifest()
    if a.pilot_per_source:
        rng = np.random.default_rng(a.seed)
        subset = []
        for group in range(5):
            source = [r for r in rows if r["source_group"] == group]
            subset.extend(
                source[i]
                for i in sorted(rng.choice(len(source), a.pilot_per_source, replace=False))
            )
        rows = subset
    config = dict(
        protocol=PROTOCOL,
        seed=a.seed,
        policy=asdict(POLICY),
        source=str(SOURCE),
        source_manifest_sha256=checksum(SOURCE / "scenes.jsonl"),
        tokens=[r["token"] for r in rows],
        construction="GT48x48/PDM32x24;0.05m;bounded interior offsets",
        threshold="strict >0.95",
        files={
            str(f.relative_to(Path(__file__).parent.parent)): checksum(f)
            for f in Path(__file__).parent.parent.rglob("*.py")
        },
    )
    run_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (a.out / "config.json").exists() and json.loads(
        (a.out / "config.json").read_text()
    ) != config:
        raise ValueError("Output configuration changed")
    write_json(a.out / "config.json", config)
    atomic_npz(
        a.out / "families.npz", coefficients=family_parameters(a.seed), family_id=np.arange(3072)
    )
    start = time.time()
    results = []
    failures = []
    counts = Counter()
    last_lists = 0.0
    last_status = 0.0

    def publish(final=False):
        nonlocal last_lists, last_status
        last_status = time.time()
        write_json(
            a.out / "status.json",
            dict(
                processed=len(results),
                expected=len(rows),
                high256=counts["high256"],
                shortfall_processed=counts["shortfall"],
                high8=counts["high8"],
                under8=counts["under8"],
                workers=a.workers,
                diverse8=counts["diverse8"],
                under_diverse8=len(results) - counts["diverse8"],
                partial_1_255=counts["partial"],
                zero_targets=counts["zero"],
                trajectory_records=len(results) * 3072,
                targets=counts["targets"],
                errors=failures,
                elapsed_s=time.time() - start,
                updated_unix=time.time(),
                complete=final and not failures and len(results) == len(rows),
            ),
        )
        if final or time.time() - last_lists >= 300:
            export_views(a.out, results)
            last_lists = time.time()

    publish()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        iterator = iter(rows)
        futures = {}

        def submit_next():
            row = next(iterator, None)
            if row is not None:
                futures[pool.submit(process_scene, (row, str(a.out), a.seed, run_hash))] = row[
                    "token"
                ]

        for _ in range(min(a.workers * 2, len(rows))):
            submit_next()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                token = futures.pop(future)
                try:
                    result = future.result()
                    results.append(result)
                    n = result["selected_count"]
                    counts["targets"] += n
                    counts["high256"] += n == 256
                    counts["shortfall"] += n < 256
                    counts["high8"] += n >= 8
                    counts["under8"] += n < 8
                    counts["diverse8"] += result["representative_count"] >= 8
                    counts["partial"] += 0 < n < 256
                    counts["zero"] += n == 0
                    print(
                        json.dumps(
                            {
                                k: result[k]
                                for k in ["token", "selected_count", "elapsed_s", "funnel"]
                            }
                        ),
                        flush=True,
                    )
                except Exception:
                    failures.append(dict(token=token, traceback=traceback.format_exc()))
                    print(json.dumps(failures[-1]), flush=True)
                submit_next()
                if (len(results) + len(failures)) % 128 == 0 or time.time() - last_status >= 30:
                    publish()
    summary = dict(
        complete=not failures and len(results) == len(rows),
        processed=len(results),
        expected=len(rows),
        high256=sum(r["selected_count"] == 256 for r in results),
        with_targets=sum(r["selected_count"] > 0 for r in results),
        high8=counts["high8"],
        under8=counts["under8"],
        diverse8=counts["diverse8"],
        under_diverse8=len(results) - counts["diverse8"],
        targets=sum(r["selected_count"] for r in results),
        errors=failures,
        elapsed_s=time.time() - start,
        scenes=results,
    )
    write_json(a.out / "summary.json", summary)
    publish(final=True)
    (a.out / "shortfall_tokens.txt").write_text(
        "".join(
            r["token"] + "\n"
            for r in sorted(results, key=lambda r: r["token"])
            if r["selected_count"] < 256
        )
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "scenes"}), flush=True)
    if failures:
        raise RuntimeError("Failed synthetic scenes retained for diagnosis")


if __name__ == "__main__":
    main()
