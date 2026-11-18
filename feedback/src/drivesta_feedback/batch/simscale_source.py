"""SimScale PDM-0..4 manifest and aligned official reference reconstruction."""

import copy
import os
import csv
import json
import lzma
import pickle
from pathlib import Path
import numpy as np
from .store import write_json, atomic, checksum


class MissingObservationHorizon(ValueError):
    """This actor cache cannot support the selected NAVSIM-v1 full TTC window."""


def require_observation_horizon(cache, required_time_s=4.9):
    obs = cache.observation
    required_index = int(round(required_time_s / obs._sample_interval))
    if len(obs._global_to_local_idcs) <= required_index:
        raise MissingObservationHorizon(
            f"Actor observations end at {(len(obs._global_to_local_idcs)-1)*obs._sample_interval:.1f}s; "
            f"NAVSIM-v1 full TTC requires {required_time_s:.1f}s. "
            "SimScale uses a different within-window TTC protocol; no implicit switch, "
            "extrapolation or TTC placeholder is permitted."
        )
    for i in range(required_index + 1):
        if obs._global_to_local_idcs[i] >= len(obs._occupancy_maps):
            raise MissingObservationHorizon("Observation mapping references missing occupancy maps")


def manifest(root, config):
    import yaml

    records, counts = [], {}
    for i, expected in enumerate(config["source_counts"]):
        name = f"synthetic_reaction_pdm_v1.0-{i}"
        filt_path = root / config["filter_root"] / (name + ".yaml")
        filt = yaml.safe_load(filt_path.read_text())
        if (filt["num_history_frames"], filt["num_future_frames"], filt["frame_interval"]) != (
            4,
            8,
            1,
        ):
            raise ValueError("Synthetic frame contract changed")
        logs, tokens = set(filt["log_names"]), set(filt["tokens"])
        invalid = (
            Path(os.environ.get("OPENSCENE_DATA_ROOT", root / "data"))
            / "sensor_blobs"
            / name
            / "invalid_tokens.yaml"
        )
        bad = (
            set(yaml.safe_load(invalid.read_text())["invalid_tokens"])
            if invalid.exists()
            else set()
        )
        cache_root = Path(os.environ.get("NAVSIM_CACHE_ROOT", root / "data/navsim_cache")) / (
            "train_metric_" + name
        )
        metadata = sorted((cache_root / "metadata").glob("*.csv"))
        if len(metadata) != 1:
            raise ValueError("Ambiguous synthetic cache metadata")
        count = 0
        for row in csv.DictReader(metadata[0].open()):
            parts = Path(row["file_name"]).parts[-4:]
            log, _, token, _ = parts
            # Invalid lists name rendered sensor samples, not current-frame tokens.
            sensor_sample = "-".join(log.split("-")[-2:])
            if log not in logs or token not in tokens or sensor_sample in bad:
                continue
            records.append(
                dict(
                    token=token,
                    log_name=log,
                    split="train",
                    source=name,
                    source_group=i,
                    sensor_sample=sensor_sample,
                    log_path=str(
                        Path(os.environ.get("OPENSCENE_DATA_ROOT", root / "data"))
                        / "navsim_logs"
                        / name
                        / (log + ".pkl")
                    ),
                    source_cache_path=str(cache_root / Path(*parts)),
                )
            )
            count += 1
        if count != expected:
            raise ValueError(f"{name}: expected {expected}, found {count}")
        counts[name] = dict(
            scenes=count,
            invalid_samples=len(bad),
            filter_digest=checksum(filt_path),
            invalid_digest=checksum(invalid) if invalid.exists() else None,
        )
    if len({r["token"] for r in records}) != len(records):
        raise ValueError("Duplicate synthetic tokens")
    return records, counts


def load_source(record, output, tail_config=None):
    from drivesta_feedback.paths import bootstrap_navsim

    root = bootstrap_navsim()
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

    install_numpy_2_pickle_compat()
    # Some older task imports installed an incomplete alias. Make numeric explicit.
    import numpy.core.numeric, sys

    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    from navsim.common.dataclasses import Scene, SensorConfig
    from .recovery_seeds import resample_recorded

    frames = pickle.load(open(record["log_path"], "rb"))
    indices = [i for i, f in enumerate(frames) if f["token"] == record["token"]]
    if len(indices) != 1 or indices[0] < 3 or len(frames) - indices[0] < 9:
        raise ValueError("Synthetic scene token/horizon mismatch")
    start = indices[0]
    selected = frames[start - 3 : start + 9]
    camera_sample = Path(selected[3]["cams"]["CAM_R0"]["data_path"]).parts[-3]
    if camera_sample != record["sensor_sample"]:
        raise ValueError("Invalid-sample exclusion and camera identity differ")
    scene = Scene.from_scene_dict_list(
        selected,
        Path(os.environ.get("OPENSCENE_DATA_ROOT", root / "data"))
        / "sensor_blobs"
        / record["source"],
        4,
        8,
        SensorConfig.build_no_sensors(),
    )
    with lzma.open(record["source_cache_path"], "rb") as f:
        original = pickle.load(f)
    require_observation_horizon(original, 4.0 if tail_config is not None else 4.9)
    pose = original.ego_state.rear_axle
    origin = np.array([pose.x, pose.y, pose.heading])
    actual = scene.frames[3].ego_status.ego_pose
    error = actual - origin
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    if (
        abs(error).max() > 1e-4
        or scene.frames[3].timestamp != original.ego_state.time_point.time_us
    ):
        raise ValueError("Synthetic scene and cache are not aligned")
    raw = scene.get_future_trajectory(num_trajectory_frames=8).poses.astype(np.float32)
    times = (np.array([f.timestamp for f in scene.frames[3:]]) - scene.frames[3].timestamp) / 1e6
    expert = resample_recorded(np.vstack([np.zeros(3), raw]), times)
    target = (
        Path(output)
        / "official_cache"
        / record["source"]
        / record["log_name"]
        / "unknown"
        / record["token"]
        / "metric_cache.pkl"
    )
    if target.exists():
        with lzma.open(target, "rb") as f:
            cache = pickle.load(f)
    else:
        from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
        from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

        scenario = NavSimScenario(
            scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
        )
        processor = MetricCacheProcessor(cache_path=None, force_feature_computation=True)
        inputs, initialization = processor._get_planner_inputs(scenario)
        processor._pdm_closed.initialize(initialization)
        trajectory = processor._pdm_closed.compute_planner_trajectory(inputs)
        planner = processor._pdm_closed
        # Preserve source simulated actors, road geometry, route and score semantics.
        if set(planner._route_lane_dict) != set(original.route_lane_ids):
            raise ValueError("Rebuilt reference route differs from synthetic source")
        old_line = getattr(original.centerline, "linestring", original.centerline)
        if planner._centerline.linestring.hausdorff_distance(old_line) > 1e-5:
            raise ValueError("Rebuilt reference centerline differs from synthetic source")
        cache = copy.copy(original)
        cache.trajectory, cache.file_path = trajectory, target
        if tail_config is not None:
            from .actor_tail import extend_observation, TailPolicy

            cache.observation, cache._alternative_observation, cache._actor_tail_meta = (
                extend_observation(original.observation, policy=TailPolicy(**tail_config))
            )
        atomic(
            target,
            lambda f: f.write(
                lzma.compress(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL), preset=0)
            ),
        )
    if tail_config is not None:
        from .actor_tail import TailPolicy
        from dataclasses import asdict

        if getattr(cache, "_actor_tail_meta", {}).get("policy") != asdict(
            TailPolicy(**tail_config)
        ):
            raise ValueError("Saved cache uses a different actor-tail policy")
    require_observation_horizon(cache)
    if (
        not hasattr(cache, "trajectory")
        or cache.ego_state.time_point != original.ego_state.time_point
    ):
        raise ValueError("Invalid completed synthetic reference cache")
    identity = dict(
        record,
        expert_gt=expert.tolist(),
        raw_expert_gt=raw.tolist(),
        timestamp_us=int(scene.frames[3].timestamp),
        ego_pose=origin.tolist(),
        official_cache_path=str(target),
        future_times_s=times.tolist(),
        source_pdm_progress=float(original.pdm_progress),
    )
    if tail_config is not None:
        identity["actor_tail"] = cache._actor_tail_meta
        identity["label_environment"] = "observed_0_4s_inferred_4_5s"
    return cache, expert, identity


def seed_paths(cache, expert):
    from nuplan.common.actor_state.state_representation import TimePoint
    from .recovery_seeds import to_local, resample_recorded
    from .recovery_motion_search import joined_reference
    from .batched_official import BatchedOfficial

    times = np.arange(41) * 0.1
    origin = cache.ego_state.rear_axle
    origin = [origin.x, origin.y, origin.heading]
    states = cache.trajectory.get_state_at_times(
        [TimePoint(cache.ego_state.time_point.time_us + round(t * 1e6)) for t in times]
    )
    reference = to_local(
        np.array([[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading] for s in states]), origin
    )
    evaluator = BatchedOfficial(cache)
    _, simulated = evaluator.score(np.array([reference[5::5], expert]))
    dyn = cache.ego_state.dynamic_car_state
    v = [dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y]
    a = [dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y]
    dense = [np.vstack([np.zeros(3), expert])]
    clocks = [np.arange(9) * 0.5]
    names = ["expert"]
    for i in range(2):
        dense.append(to_local(simulated[i, :, :3], origin))
        clocks.append(times)
        names.append("tracked_" + str(i))
    for join in (2.0, 3.0, 4.0):
        dense.append(joined_reference(times, reference, v, a, join))
        clocks.append(times)
        names.append("joined_" + str(join))
    paths, metadata = [], []
    for parent, clock, name in zip(dense, clocks, names):
        for fraction in (1.0, 0.98, 0.95, 0.9, 0.85):
            t = np.arange(1, 9) * 0.5
            desired = t - (1 - fraction) * t * t / 4
            path = resample_recorded(parent, clock, desired)
            if not any(np.array_equal(path, p) for p in paths):
                paths.append(path)
                metadata.append(dict(parent=name, time_fraction=fraction))
    return np.array(paths), metadata, evaluator, origin, v, a
