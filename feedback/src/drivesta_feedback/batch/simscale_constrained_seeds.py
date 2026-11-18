"""Constrained official rollout search for SimScale recovery seeds.

Optimization aids never replace the original admission predicate. Every queried
path and both official label vectors are retained, including infeasible probes.
"""

from pathlib import Path
import numpy as np

from .store import write_npz, write_json
from .simscale_dataset import path_key
from .physical_sampling import physical_mask, score_mask
from .filter_targets import tracking_errors, wrap
from .adaptive_sampling import diverse_indices, quality
from .recovery_projection_search import motion_slack
from .recovery_rollout_search import (
    Geometry,
    spline_correction,
    comfort_slack,
    direction_slack,
    SearchStop,
)


def exclude_expert_probes(arrays, expert, raw_expert):
    """Retain GT probe scores, but do not count unchanged GT as a new target."""
    forbidden = {path_key(expert), path_key(raw_expert)}
    arrays["eligible"] = np.asarray(arrays["eligible"], bool).copy()
    for i, t in enumerate(arrays["trajectories"]):
        if path_key(t) in forbidden:
            arrays["eligible"][i] = False
    return arrays


def optimize_seeds(
    out,
    cfg,
    evaluator,
    space,
    expert,
    policy,
    velocity,
    acceleration,
    origin,
    old,
    parents,
    raw_expert,
):
    from scipy.optimize import minimize

    path = Path(out) / "optimized_seeds.npz"
    report = Path(out) / "optimized_seeds.json"
    settings = cfg["constrained_seeds"]
    geometry = Geometry.__new__(Geometry)
    geometry.cache = space.cache
    geometry.road = space.road
    geometry.route = space.route
    geometry.vehicle = space.vehicle
    geometry.sampling = space.sampling
    limits = np.array(
        [
            policy.max_tracking_ade_m,
            policy.max_tracking_error_m,
            policy.max_tracking_heading_error_rad,
        ]
    )
    columns = ("trajectories", "labels", "alternative_labels", "tracking", "eligible")
    arrays = {k: [] for k in columns}
    slacks = []
    objectives = []
    memo = {}
    runs = []
    forbidden = {path_key(expert), path_key(raw_expert)}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if z["digest"].tobytes().decode() != cfg["digest"]:
                raise ValueError("Optimizer configuration changed")
            arrays = {k: list(z[k]) for k in columns}
            slacks = list(z["slacks"])
            objectives = list(z["objectives"])
            for i, t in enumerate(arrays["trajectories"]):
                memo[path_key(t)] = (objectives[i], slacks[i])
            arrays["eligible"] = [
                bool(v) and path_key(t) not in forbidden
                for t, v in zip(arrays["trajectories"], arrays["eligible"])
            ]
            if bool(z["finished"]):
                return exclude_expert_probes(
                    {k: np.asarray(v) for k, v in arrays.items()}, expert, raw_expert
                )

    def save(finished=False):
        write_npz(
            path,
            **{k: np.asarray(v) for k, v in arrays.items()},
            slacks=np.asarray(slacks),
            objectives=np.asarray(objectives),
            finished=np.array(finished),
            digest=np.frombuffer(cfg["digest"].encode(), np.uint8),
        )

    def query(flat):
        t = np.asarray(flat, np.float32).reshape(8, 3).copy()
        t[:, 2] = wrap(t[:, 2])
        key = path_key(t)
        if key in memo:
            return memo[key]
        if (
            len(arrays["labels"]) >= settings["max_queries"]
            or sum(arrays["eligible"]) >= settings["seed_goal"]
        ):
            raise SearchStop()
        y, sim = evaluator.score(t[None])
        alt = evaluator.alternative_labels.copy()
        errors = np.asarray(tracking_errors(t, sim[0], origin))
        physical, _ = physical_mask(t[None], expert, policy, velocity, acceleration)
        near = np.linalg.norm(t[:, :2] - expert[:, :2], axis=1).max() <= cfg["max_gt_deviation_m"]
        good = bool(
            key not in forbidden
            and physical[0]
            and near
            and score_mask(y, policy)[0]
            and score_mask(alt, policy)[0]
            and (errors <= limits).all()
            and space(t)
        )
        planned = geometry.clearance(geometry.states(t))
        tracked = geometry.clearance(sim[0])
        # Inward margin only guides search; no boundary is dilated.
        margin = (
            np.tile(np.minimum(np.arange(41) * 0.1 / 0.5, 1.0), 2) * settings["spatial_margin_m"]
        )
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            BBCoordsIndex,
            EgoAreaIndex,
        )

        scorer = evaluator.runtime.scorer
        direction = direction_slack(
            scorer._ego_coords[1, :, BBCoordsIndex.CENTER],
            scorer._ego_areas[1, :, EgoAreaIndex.ONCOMING_TRAFFIC],
            horizon=int(scorer._config.driving_direction_horizon / 0.1),
            threshold_m=scorer._config.driving_direction_compliance_threshold,
        )
        slack = np.r_[
            motion_slack(t, velocity, acceleration, policy),
            planned - margin,
            tracked[:41] - margin[:41],
            direction,
            1 - errors / (limits * 0.98),
            comfort_slack(sim[0]),
            cfg["max_gt_deviation_m"] - np.linalg.norm(t[:, :2] - expert[:, :2], axis=1),
        ]
        # Progress supplies a continuous signal even when PDM's safety product is zero.
        value = (
            -float(scorer._progress_raw[1]) / max(float(scorer._progress_raw[0]), 1.0)
            + 0.1 * np.maximum(-tracked, 0).mean()
        )
        row = dict(
            trajectories=t, labels=y[0], alternative_labels=alt[0], tracking=errors, eligible=good
        )
        for k in columns:
            arrays[k].append(row[k])
        slacks.append(slack)
        objectives.append(value)
        memo[key] = (value, slack)
        if len(arrays["labels"]) % 32 == 0:
            save()
        return value, slack

    priorities = quality(
        old["labels"], old["alternative_labels"], old["tracking"], limits, old["eligible"]
    )
    ids = diverse_indices(old["trajectories"], priorities, 12, 0.25)
    options = np.concatenate([old["trajectories"][ids], parents[:4]])
    ranked = []
    for t in options:
        if np.linalg.norm(t[:, :2] - expert[:, :2], axis=1).max() > cfg["max_gt_deviation_m"]:
            continue
        if any(np.linalg.norm(t[:, :2] - p[:, :2], axis=1).mean() < 0.1 for _, p in ranked):
            continue
        clearance = geometry.clearance(geometry.states(t))
        motion = motion_slack(t, velocity, acceleration, policy)
        penalty = np.maximum(-clearance, 0).mean() + np.maximum(-motion, 0).mean()
        ranked.append((penalty, t))
    try:
        for parent_id, (_, base) in enumerate(
            sorted(ranked, key=lambda x: x[0])[: settings["parents"]]
        ):
            base = base.astype(float)
            base[:, 2] = np.unwrap(np.r_[0.0, base[:, 2]])[1:]
            transform = lambda x: spline_correction(base, x).ravel()
            width = settings["position_bound_m"]
            bounds = list(zip(np.tile([-width, -width, -4.5], 4), np.tile([width, width, 4.5], 4)))
            res = minimize(
                lambda x: query(transform(x))[0],
                np.zeros(12),
                method="SLSQP",
                bounds=bounds,
                constraints=[dict(type="ineq", fun=lambda x: query(transform(x))[1])],
                options=dict(maxiter=settings["iterations"], ftol=1e-6, eps=0.002),
            )
            query(transform(res.x))
            runs.append(
                dict(
                    parent=parent_id,
                    success=bool(res.success),
                    message=str(res.message),
                    iterations=int(res.nit),
                )
            )
    except SearchStop:
        runs.append(dict(stopped="query budget or verified seed goal"))
    finally:
        # On failure keep the archive resumable and do not declare completion.
        save(False)
    save(True)
    write_json(
        report,
        dict(
            queries=len(arrays["labels"]),
            eligible=int(sum(arrays["eligible"])),
            runs=runs,
            original_admission_unchanged=True,
            all_probes_retained=True,
        ),
    )
    return {k: np.asarray(v) for k, v in arrays.items()}
