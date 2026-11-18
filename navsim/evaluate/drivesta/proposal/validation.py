"""Official reference-plus-one-candidate scoring for held-out predictions."""

import lzma
from pathlib import Path
import pickle

import numpy as np

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.drivesta.losses.proposal import COMPONENTS


class OfficialPDM:
    def __init__(self, cache_root, records):
        self.paths = {
            r["token"]: Path(cache_root)
            / r["log_name"]
            / "unknown"
            / r["token"]
            / "metric_cache.pkl"
            for r in records
        }
        missing = [token for token, path in self.paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Complete official metric caches missing: {missing[:8]}"
            )
        self.sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.sampling)
        self.scorer = PDMScorer(self.sampling, PDMScorerConfig())

    def score(self, token, trajectories):
        with lzma.open(self.paths[token], "rb") as stream:
            cache = pickle.load(stream)
        if not hasattr(cache, "trajectory"):
            raise ValueError(
                "Validation requires the complete official reference trajectory cache"
            )
        rows = []
        for trajectory in np.asarray(trajectories):
            result = pdm_score(
                cache,
                Trajectory(trajectory),
                self.sampling,
                self.simulator,
                self.scorer,
            )
            rows.append([float(getattr(result, key)) for key in (*COMPONENTS, "score")])
        labels = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(labels).all() or (labels < 0).any() or (labels > 1).any():
            raise ValueError(f"Invalid official result for {token}")
        return labels
