"""Frame-local corrected NAVSIM-v2 labels; cross-frame EC is handled by the caller.

Uses the official simulator and component scorer, with only proposal-progress
normalization changed to independent (PDM-reference, candidate) pairs. This
must match separate official pdm_score calls before a corpus run is accepted.
"""

import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import sys, json, time, lzma, pickle
from pathlib import Path

R = Path(os.environ.get("DRIVESTA_ROOT", str(Path(__file__).resolve().parents[4])))
os.environ.setdefault(
    "NUPLAN_MAPS_ROOT", str(Path(os.environ.get("OPENSCENE_DATA_ROOT", str(R / "data"))) / "maps")
)
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
from navsim.v2.pickle_compat import load as load_v2
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.v2.common.dataclasses import Trajectory
from navsim.v2.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
from navsim.v2.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.v2.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.v2.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.v2.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents

SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
OUTPUT_SAMPLING = TrajectorySampling(num_poses=8, interval_length=0.5)
FIELDS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
]


class IndependentPairScorer(PDMScorer):
    def _aggregate_pdm_scores(self):
        # Candidate 0 is the same PDM reference used in each official 2-plan call.
        multi = self._multi_metrics.prod(axis=0)
        masked = self._progress_raw * multi
        denominator = np.maximum(masked[0], masked)
        progress = np.ones_like(denominator)
        active = denominator > self._config.progress_distance_threshold
        progress[active] = np.clip(self._progress_raw[active] / denominator[active], 0, 1)
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = progress
        w = self._config.weighted_metrics_array.copy()
        w[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = 0
        return multi * (self._weighted_metrics * w[:, None]).sum(axis=0) / w.sum()


def component_rows(rows):
    return np.array([[float(row.at[0, k]) for k in FIELDS] for row in rows], dtype=np.float64)


def score_from_components(c, ec=None):
    product = c[..., :4].prod(axis=-1)
    numer = (c[..., 4:] * np.array([5.0, 5.0, 2.0, 2.0])).sum(axis=-1)
    return product * (numer / 14 if ec is None else (numer + 2 * np.asarray(ec)) / 16)


class ArchiveScorer:
    def __init__(self, cache):
        self.cache = cache
        self.sim = PDMSimulator(SAMPLING)
        self.scorer = IndependentPairScorer(SAMPLING, PDMScorerConfig(human_penalty_filter=True))
        self.official = PDMScorer(SAMPLING, PDMScorerConfig(human_penalty_filter=True))
        self.traffic = LogReplayTrafficAgents(SAMPLING)
        self.ref = get_trajectory_as_array(cache.trajectory, SAMPLING, cache.ego_state.time_point)
        human = get_trajectory_as_array(
            transform_trajectory(cache.human_trajectory, cache.ego_state),
            SAMPLING,
            cache.ego_state.time_point,
        )
        human_states = self.sim.simulate_proposals(human[None], cache.ego_state)
        self.environment = self.traffic.simulate_environment(human_states[0], cache)
        # Match official filter: human is scored alone, without history argument.
        h = self.official.score_proposals(
            human_states,
            cache.observation,
            cache.centerline,
            cache.route_lane_ids,
            cache.drivable_area_map,
            cache.map_parameters,
            self.environment,
        )[0]
        self.filter_mask = component_rows([h])[0] == 0
        self.expert_row, self.expert_states = pdm_score(
            cache, cache.human_trajectory, SAMPLING, self.sim, self.official, self.traffic
        )
        self.expert_components = component_rows([self.expert_row])[0]

    def score(self, trajectories, chunk=128, return_states=False):
        c = self.cache
        raw = []
        states_all = []
        for start in range(0, len(trajectories), chunk):
            arrays = [self.ref]
            for t in trajectories[start : start + chunk]:
                arrays.append(
                    get_trajectory_as_array(
                        transform_trajectory(
                            Trajectory(np.asarray(t, dtype=np.float64), OUTPUT_SAMPLING),
                            c.ego_state,
                        ),
                        SAMPLING,
                        c.ego_state.time_point,
                    )
                )
            states = self.sim.simulate_proposals(np.stack(arrays), c.ego_state)
            rows = self.scorer.score_proposals(
                states,
                c.observation,
                c.centerline,
                c.route_lane_ids,
                c.drivable_area_map,
                c.map_parameters,
                self.environment,
                c.past_human_trajectory,
            )
            raw.append(component_rows(rows[1:]))
            if return_states:
                states_all.append(states[1:])
        raw = np.concatenate(raw)
        filtered = raw.copy()
        filtered[:, self.filter_mask] = 1
        assert np.isfinite(filtered).all() and filtered.min() >= 0 and filtered.max() <= 1
        return raw, filtered, np.concatenate(states_all) if return_states else None


def probe(cache_path, bank_path, out):
    with lzma.open(cache_path, "rb") as f:
        c = load_v2(f)
    a = np.load(bank_path, mmap_mode="r")
    engine = ArchiveScorer(c)
    ids = np.unique(np.r_[np.arange(8), np.linspace(0, len(a) - 1, 16, dtype=int)])
    t = time.time()
    raw, batched, states = engine.score(a["trajectories"][ids], chunk=8, return_states=True)
    rows = []
    official_states = []
    for traj in a["trajectories"][ids]:
        row, s = pdm_score(
            c,
            Trajectory(traj.astype(np.float64), OUTPUT_SAMPLING),
            SAMPLING,
            engine.sim,
            engine.official,
            engine.traffic,
        )
        rows.append(row)
        official_states.append(s)
    ref = component_rows(rows)
    error = float(np.max(np.abs(ref - batched)))
    state_error = float(np.max(np.abs(np.stack(official_states) - states)))
    assert error < 1e-10, (error, ref, batched)
    assert state_error < 1e-10, state_error
    _, other, _ = engine.score(a["trajectories"][ids][::-1], chunk=5)
    assert np.max(np.abs(other[::-1] - batched)) < 1e-10
    checks_s = time.time() - t
    t = time.time()
    raw, filtered, _ = engine.score(a["trajectories"], chunk=128)
    report = {
        "token": Path(bank_path).parent.name,
        "official_comparisons": len(ids),
        "max_component_error": error,
        "max_simulation_error": state_error,
        "chunk_order_invariance": True,
        "checks_seconds": checks_s,
        "all3072_seconds": time.time() - t,
        "expert_components": engine.expert_components.tolist(),
        "filter_mask": engine.filter_mask.tolist(),
        "mean_frame_local_score": float(score_from_components(filtered).mean() * 100),
        "ge95_without_EC": int((score_from_components(filtered) >= 0.95 - 1e-8).sum()),
        "warning": "Frame-local diagnostic; NOT full navtest EPDMS. EC pairing pending.",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "components.npz",
        raw=raw,
        filtered=filtered,
        fields=np.array(FIELDS),
        expert=engine.expert_components,
        ec_lower=score_from_components(filtered, 0),
        ec_upper=score_from_components(filtered, 1),
    )
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
