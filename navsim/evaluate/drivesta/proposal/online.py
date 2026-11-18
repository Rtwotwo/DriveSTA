"""Batched simulation with the official reference-plus-one score semantics."""

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import lzma
import multiprocessing as mp
from pathlib import Path
import pickle

import numpy as np


@lru_cache(maxsize=8)
def load_cache(path):
    with lzma.open(path, "rb") as stream:
        return pickle.load(stream)


@lru_cache(maxsize=1)
def runtime():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
        PDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
        PDMSimulator,
    )

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    return sampling, PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())


def score_scene(job):
    path, trajectories = job
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex as M,
        WeightedMetricIndex as W,
    )

    cache = load_cache(path)
    sampling, simulator, scorer = runtime()
    ego = cache.ego_state
    arrays = [get_trajectory_as_array(cache.trajectory, sampling, ego.time_point)]
    arrays.extend(
        get_trajectory_as_array(
            transform_trajectory(Trajectory(t), ego), sampling, ego.time_point
        )
        for t in np.asarray(trajectories, dtype=np.float64)
    )
    simulated = simulator.simulate_proposals(np.stack(arrays), ego)
    scorer.score_proposals(
        simulated,
        cache.observation,
        cache.centerline,
        cache.route_lane_ids,
        cache.drivable_area_map,
    )
    # Only progress aggregation couples candidates. Recompute its denominator
    # separately for each candidate and the SAME official reference trajectory.
    multi = scorer._multi_metrics.prod(axis=0)
    raw = scorer._progress_raw * multi
    denominator = np.maximum(raw[0], raw[1:])
    progress = np.where(
        denominator > scorer._config.progress_distance_threshold,
        raw[1:] / np.maximum(denominator, 1e-12),
        (multi[1:] != 0).astype(float),
    )
    weighted = scorer._weighted_metrics[:, 1:].copy()
    weighted[W.PROGRESS] = progress
    weights = scorer._config.weighted_metrics_array
    total = multi[1:] * (weighted * weights[:, None]).sum(0) / weights.sum()
    labels = np.stack(
        [
            scorer._multi_metrics[M.NO_COLLISION, 1:],
            scorer._multi_metrics[M.DRIVABLE_AREA, 1:],
            progress,
            weighted[W.TTC],
            weighted[W.COMFORTABLE],
            weighted[W.DRIVING_DIRECTION],
            total,
        ],
        -1,
    )
    if not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
        raise ValueError(f"Invalid official online labels: {path}")
    return labels


class OnlinePDM:
    def __init__(self, cache_root, records, workers=4):
        self.paths = {
            r["token"]: str(
                Path(cache_root)
                / r["log_name"]
                / "unknown"
                / r["token"]
                / "metric_cache.pkl"
            )
            for r in records
        }
        self.workers = int(workers)
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        self.pool = None

    def score(self, tokens, trajectories):
        jobs = [
            (self.paths[token], trajectory)
            for token, trajectory in zip(tokens, trajectories)
        ]
        if len(jobs) != len(tokens) or len(tokens) != len(trajectories):
            raise ValueError("Online scoring batch size mismatch")
        if not self.workers:
            return np.stack([score_scene(job) for job in jobs])
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                self.workers, mp_context=mp.get_context("spawn")
            )
        return np.stack(list(self.pool.map(score_scene, jobs)))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
