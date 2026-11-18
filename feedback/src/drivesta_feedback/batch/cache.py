"""Build complete official caches beside feedback data; preserve training caches."""

import lzma
import os
from pathlib import Path
import pickle
import tempfile

from .store import atomic, lock


def official_cache(scene, root, protocol):
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    scenario = NavSimScenario(
        scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
    )
    target = (
        root
        / protocol["scene_scope"]["official_cache_root"]
        / scenario.log_name
        / scenario.scenario_type
        / scenario.token
        / "metric_cache.pkl"
    )

    def read(path):
        with lzma.open(path, "rb") as f:
            cache = pickle.load(f)
        for key in (
            "trajectory",
            "ego_state",
            "observation",
            "centerline",
            "route_lane_ids",
            "drivable_area_map",
        ):
            if not hasattr(cache, key):
                raise ValueError("Incomplete official cache: " + key)
        return cache

    with lock(target.parent):
        if target.exists():
            return read(target), target
        with tempfile.TemporaryDirectory(prefix=".build-", dir=target.parent) as temp:
            processor = MetricCacheProcessor(cache_path=temp, force_feature_computation=True)
            entry = processor.compute_metric_cache(scenario)
            cache = read(Path(entry.file_name))
            cache.file_path = target
            payload = lzma.compress(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL), preset=0)
            atomic(target, lambda f: f.write(payload), read)
        return read(target), target
