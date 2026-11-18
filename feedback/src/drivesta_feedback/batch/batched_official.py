"""Vectorized tracking; official reference-plus-single-candidate PDM labels."""

import numpy as np


class BatchedOfficial:
    def __init__(self, cache):
        from .source import Evaluator

        self.runtime = Evaluator(cache)
        self.cache = cache

    def score(self, paths):
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            MultiMetricIndex as M,
            WeightedMetricIndex as W,
        )

        run, cache = self.runtime, self.cache
        arrays = [
            get_trajectory_as_array(cache.trajectory, run.sampling, cache.ego_state.time_point)
        ]
        arrays += [
            get_trajectory_as_array(
                transform_trajectory(Trajectory(t), cache.ego_state),
                run.sampling,
                cache.ego_state.time_point,
            )
            for t in paths
        ]
        simulated = run.simulator.simulate_proposals(np.stack(arrays), cache.ego_state)
        scorer = run.scorer
        scorer.score_proposals(
            simulated,
            cache.observation,
            cache.centerline,
            cache.route_lane_ids,
            cache.drivable_area_map,
        )
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
            raise ValueError("Invalid official labels")
        self.alternative_labels = labels.copy()
        if hasattr(cache, "_alternative_observation"):
            # All states and non-TTC components are fixed. Both actor tails have
            # the exact same measured prefix; only the TTC projection can differ.
            scorer._observation = cache._alternative_observation
            try:
                scorer._calculate_ttc()
                self.alternative_labels[:, 3] = scorer._weighted_metrics[W.TTC, 1:]
            finally:
                scorer._observation = cache.observation
            y = self.alternative_labels
            y[:, 6] = y[:, 0] * y[:, 1] * (5 * y[:, 2] + 5 * y[:, 3] + 2 * y[:, 4]) / 12
        return labels, simulated[1:].copy()
