"""Scene/cache alignment, exact TTO construction and official singleton scoring."""

from dataclasses import asdict
from functools import lru_cache
import lzma
import pickle
import numpy as np
import torch

from drivesta_feedback.paths import bootstrap_navsim

ROOT = bootstrap_navsim()
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.tools.trajectory_mining.path_endpoint_surface import _build_warp
from navsim.evaluate.pdm_score import pdm_score, pdm_score_with_details
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

COMPONENTS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
]


@lru_cache(maxsize=2)
def loader(log, tokens, log_root):
    filt = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        log_names=[log],
        tokens=list(tokens),
        max_scenes=None,
    )
    return SceneLoader(
        ROOT / log_root, ROOT / "data/sensor_blobs", filt, SensorConfig.build_no_sensors()
    )


def load_scene(record, job_tokens, protocol):
    scenes = loader(record["log_name"], tuple(job_tokens), protocol["scene_scope"]["log_root"])
    scene = scenes.get_scene_from_token(record["token"])
    gt = scene.get_future_trajectory(num_trajectory_frames=8).poses.astype(np.float64)
    cache_path = (
        ROOT / protocol["scene_scope"]["metric_cache_root"] / record["metric_cache_relative_path"]
    )
    with lzma.open(cache_path, "rb") as f:
        cache = pickle.load(f)
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    pose = np.asarray(frame.ego_status.ego_pose)
    cached_pose = np.array(
        [
            cache.ego_state.rear_axle.x,
            cache.ego_state.rear_axle.y,
            cache.ego_state.rear_axle.heading,
        ]
    )
    error = pose - cached_pose
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    if gt.shape != (8, 3) or not np.isfinite(gt).all():
        raise ValueError("Invalid expert trajectory")
    if abs(error).max() >= 1e-4 or int(frame.timestamp) != cache.ego_state.time_point.time_us:
        raise ValueError("Scene/cache timestamp or pose mismatch")
    source_cache_path = cache_path
    if not hasattr(cache, "trajectory"):
        from .cache import official_cache

        cache, cache_path = official_cache(scene, ROOT, protocol)
        rebuilt_pose = np.array(
            [
                cache.ego_state.rear_axle.x,
                cache.ego_state.rear_axle.y,
                cache.ego_state.rear_axle.heading,
            ]
        )
        if not np.allclose(
            rebuilt_pose, cached_pose, atol=1e-4, rtol=0
        ) or cache.ego_state.time_point.time_us != int(frame.timestamp):
            raise ValueError("Rebuilt official cache is not aligned")
    metadata = dict(
        record,
        timestamp_us=int(frame.timestamp),
        ego_pose=cached_pose.tolist(),
        pose_error=float(abs(error).max()),
        expert_gt=gt.tolist(),
        source_cache_path=str(source_cache_path),
        official_cache_path=str(cache_path),
        pdm_config=asdict(PDMScorerConfig()),
        component_names=COMPONENTS,
    )
    return scene, gt, cache, metadata


def construct(gt, keys, protocol):
    goals = gt[-1, :2] + np.asarray(keys) * protocol["sampling"]["integer_lattice_m"]
    n = len(keys)
    with torch.no_grad():
        paths, _, _ = _build_warp(
            torch.from_numpy(gt.astype(np.float32))[None],
            torch.ones(n, 1),
            torch.zeros(n, dtype=torch.long),
            torch.from_numpy(goals.astype(np.float32)),
            torch.ones(n),
            torch.float32,
        )
    paths = paths.numpy()
    error = float(np.linalg.norm(paths[:, -1, :2] - goals, axis=1).max())
    first = float(np.linalg.norm(paths[:, 0, :2] - gt[0, :2], axis=1).max())
    tolerance = protocol["constructor"]["max_planned_endpoint_error_m"]
    if not np.isfinite(paths).all() or max(error, first) >= tolerance:
        raise ValueError("Trajectory construction exceeds tolerance")
    return paths, error


class Evaluator:
    def __init__(self, cache):
        self.cache = cache
        self.sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.sampling)
        self.scorer = PDMScorer(self.sampling, PDMScorerConfig())

    def score(self, trajectory, details=False):
        fn = pdm_score_with_details if details else pdm_score
        result = fn(self.cache, Trajectory(trajectory), self.sampling, self.simulator, self.scorer)
        if details:
            result, extra = result
        values = np.array([float(getattr(result, k)) for k in COMPONENTS], dtype=np.float64)
        if not np.isfinite(values).all() or (values < -1e-8).any() or (values > 1 + 1e-8).any():
            raise ValueError("Invalid official PDM result")
        return (values, extra["simulated_states_pred"]) if details else values
