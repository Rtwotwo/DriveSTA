"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
import torch
from navsim.tools.trajectory_mining.config import TTOConfig
from navsim.tools.trajectory_mining.endpoint_surface import _wrap, _mixture_trials, _neighborhoods, _surface
from typing import Sequence, Callable
from dataclasses import dataclass

PathEndpointScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PathEndpointResult:
    candidates: torch.Tensor
    available: torch.Tensor
    parents: torch.Tensor
    details: dict[str, torch.Tensor]


def _quintic_progress(steps: int, *, device: torch.device) -> torch.Tensor:
    """Minimum-jerk boundary blend with zero first/second derivative at ends."""
    time = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)
    return 10 * time.pow(3) - 15 * time.pow(4) + 6 * time.pow(5)


def _history_summary(
    history: torch.Tensor, history_dt: float
) -> dict[str, torch.Tensor]:
    """Extract the deployed boundary and three-frame finite-difference audit."""
    if history.ndim != 2 or history.shape[0] < 3 or history.shape[1] < 7:
        raise ValueError("ego_history must have shape [H>=3,F>=7]")
    if not torch.isfinite(history[:, :7]).all():
        raise ValueError("ego_history pose/velocity/acceleration must be finite")
    pose = history[-3:, :3].float()
    velocity = history[-1, 3:5].float()
    acceleration = history[-1, 5:7].float()
    pose_velocity = (pose[-1, :2] - pose[-2, :2]) / history_dt
    pose_acceleration = (pose[-1, :2] - 2 * pose[-2, :2] + pose[-3, :2]) / history_dt**2
    yaw_rates = _wrap(pose[1:, 2] - pose[:-1, 2]) / history_dt
    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "pose_velocity": pose_velocity,
        "pose_acceleration": pose_acceleration,
        "past_yaw_rate": yaw_rates[-1],
        "past_yaw_acceleration": (yaw_rates[-1] - yaw_rates[-2]) / history_dt,
    }


def _parent_variants(
    proposals: torch.Tensor,
    support: torch.Tensor,
    local_weights: torch.Tensor,
    count: int,
    min_ade: float,
) -> list[int]:
    """Choose high-weight, full-path-distinct parents for one requested goal."""
    order = torch.argsort(local_weights, descending=True, stable=True).tolist()
    selected: list[int] = []
    for local in order:
        candidate = int(support[local])
        if selected:
            ade = torch.linalg.vector_norm(
                proposals[selected, :, :2] - proposals[candidate, :, :2], dim=-1
            ).mean(-1)
            if bool((ade < min_ade).any()):
                continue
        selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def _build_warp(
    proposals: torch.Tensor,
    weights: torch.Tensor,
    parents: torch.Tensor,
    requested_goals: torch.Tensor,
    factors: torch.Tensor,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a proposal-manifold path with endpoint/history null residual.

    ``weights`` supplies a convex seed trajectory.  A parent-minus-seed residual
    supplies a distinct full-path mode.  Removing the residual's quintic
    boundary interpolation projects it into the null space of first-position
    and endpoint constraints.  Different parents therefore give different
    middle paths with exactly the same achieved terminal point.
    """
    parent = proposals[parents].float()
    parent_goals = parent[:, -1, :2]
    parent_one_hot = torch.zeros_like(weights)
    parent_one_hot[torch.arange(weights.shape[0], device=weights.device), parents] = 1.0
    achieved_weights = (
        factors[:, None] * weights + (1.0 - factors[:, None]) * parent_one_hot
    )
    seed_xy = torch.einsum("qn,ntd->qtd", achieved_weights, proposals[..., :2])
    achieved_goals = parent_goals + factors[:, None] * (requested_goals - parent_goals)
    seed_xy[:, -1] = achieved_goals
    blend = _quintic_progress(parent.shape[1], device=parent.device)
    residual = parent[..., :2] - seed_xy
    boundary = (1.0 - blend)[None, :, None] * residual[:, :1] + blend[
        None, :, None
    ] * residual[:, -1:, :]
    xy = seed_xy + residual - boundary
    xy[:, 0] = seed_xy[:, 0]
    xy[:, -1] = achieved_goals
    reference_heading = torch.atan2(
        torch.einsum("qn,nt->qt", achieved_weights, torch.sin(proposals[..., 2])),
        torch.einsum("qn,nt->qt", achieved_weights, torch.cos(proposals[..., 2])),
    )
    full = torch.cat((torch.zeros_like(xy[:, :1]), xy), dim=1)
    tangent = torch.cat(
        (full[:, 2:] - full[:, :-2], full[:, -1:] - full[:, -2:-1]), dim=1
    )
    heading = torch.atan2(tangent[..., 1], tangent[..., 0])
    heading = torch.where(
        torch.linalg.vector_norm(tangent, dim=-1) > 0.02, heading, reference_heading
    )
    trajectory = torch.cat((xy, _wrap(heading)[..., None]), dim=-1).to(output_dtype)
    trajectory[:, -1, :2] = achieved_goals.to(output_dtype)
    return (trajectory, achieved_goals, achieved_weights)


def _path_metrics(
    trajectory: torch.Tensor,
    parent: torch.Tensor,
    requested_goal: torch.Tensor,
    achieved_goal: torch.Tensor,
    history: dict[str, torch.Tensor],
    config: TTOConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard validity, rejection bits, metrics, and soft dynamics cost.

    Metric columns are max speed, vector acceleration, vector jerk, yaw rate,
    lateral acceleration, yaw acceleration, parent XY deviation, parent heading
    deviation, output endpoint error, request/achievement consensus error,
    FP32-to-output endpoint quantization, and first-interval velocity mismatch.
    """
    actual = trajectory.float()
    parent = parent.float()
    zero_xy = torch.zeros_like(actual[:, :1, :2])
    full_xy = torch.cat((zero_xy, actual[..., :2]), dim=1)
    dt = config.trajectory_dt
    velocity = torch.diff(full_xy, dim=1) / dt
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    initial_velocity = history["velocity"][None, None].expand(trajectory.shape[0], 1, 2)
    acceleration = (
        torch.diff(torch.cat((initial_velocity, velocity), dim=1), dim=1) / dt
    )
    initial_acceleration = history["acceleration"][None, None].expand(
        trajectory.shape[0], 1, 2
    )
    jerk = (
        torch.diff(torch.cat((initial_acceleration, acceleration), dim=1), dim=1) / dt
    )
    zero_heading = torch.zeros_like(actual[:, :1, 2])
    yaw_rate = (
        _wrap(torch.diff(torch.cat((zero_heading, actual[..., 2]), dim=1), dim=1)) / dt
    )
    past_yaw = history["past_yaw_rate"].reshape(1, 1).expand(trajectory.shape[0], 1)
    yaw_acceleration = torch.diff(torch.cat((past_yaw, yaw_rate), dim=1), dim=1) / dt
    goal_cast = achieved_goal.to(trajectory.dtype).float()
    metrics = torch.stack(
        (
            speed.amax(-1),
            torch.linalg.vector_norm(acceleration, dim=-1).amax(-1),
            torch.linalg.vector_norm(jerk, dim=-1).amax(-1),
            yaw_rate.abs().amax(-1),
            (speed * yaw_rate.abs()).amax(-1),
            yaw_acceleration.abs().amax(-1),
            torch.linalg.vector_norm(actual[..., :2] - parent[..., :2], dim=-1).amax(
                -1
            ),
            _wrap(actual[..., 2] - parent[..., 2]).abs().amax(-1),
            torch.linalg.vector_norm(actual[:, -1, :2] - goal_cast, dim=-1),
            torch.linalg.vector_norm(actual[:, -1, :2] - requested_goal, dim=-1),
            torch.linalg.vector_norm(actual[:, -1, :2] - achieved_goal, dim=-1),
            torch.linalg.vector_norm(
                velocity[:, 0] - history["velocity"][None], dim=-1
            ),
        ),
        dim=-1,
    )
    limits = torch.tensor(
        (
            config.endpoint_max_speed,
            config.endpoint_max_accel,
            config.endpoint_max_jerk,
            config.endpoint_max_yaw_rate,
            config.endpoint_max_lateral_accel,
            config.path_max_yaw_accel,
            config.endpoint_local_max_xy,
            config.endpoint_heading_tolerance,
            0.0,
        ),
        device=actual.device,
    )
    bad = metrics[:, :9] > limits[None]
    finite = torch.isfinite(actual).flatten(1).all(-1) & torch.isfinite(metrics).all(-1)
    bit_weights = 2 ** torch.arange(9, device=actual.device, dtype=torch.long)
    reasons = (bad.long() * bit_weights).sum(-1) + (~finite).long() * 512
    ratios = metrics[:, :6] / limits[:6][None]
    dynamics_cost = ratios.square().mean(-1)
    return (finite & ~bad.any(-1), reasons, metrics, dynamics_cost)


def _complete_path_variants(
    proposals: torch.Tensor,
    weights: torch.Tensor,
    parents: torch.Tensor,
    requested_goals: torch.Tensor,
    config: TTOConfig,
    output_dtype: torch.dtype,
    ego_history: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Complete paths and project infeasible goals toward their local parents."""
    count = weights.shape[0]
    history = _history_summary(ego_history, config.history_dt)
    selected = torch.zeros(
        count, proposals.shape[1], 3, device=proposals.device, dtype=output_dtype
    )
    selected_metrics = torch.zeros(count, 12, device=proposals.device)
    selected_reasons = torch.zeros(count, device=proposals.device, dtype=torch.long)
    selected_cost = torch.zeros(count, device=proposals.device)
    selected_factor = torch.zeros(count, device=proposals.device)
    selected_goal = proposals[parents, -1, :2].clone()
    selected_achieved_weights = torch.zeros_like(weights)
    valid = torch.zeros(count, device=proposals.device, dtype=torch.bool)
    full_valid = torch.zeros_like(valid)
    for step in range(config.path_projection_steps):
        factor = config.path_projection_shrink**step
        factors = torch.full((count,), factor, device=proposals.device)
        candidate, achieved, achieved_weights = _build_warp(
            proposals, weights, parents, requested_goals, factors, output_dtype
        )
        ok, reasons, metrics, cost = _path_metrics(
            candidate, proposals[parents], requested_goals, achieved, history, config
        )
        if step == 0:
            full_valid = ok.clone()
        take = ~valid & ok
        selected[take] = candidate[take]
        selected_metrics[take] = metrics[take]
        selected_reasons[take] = reasons[take]
        selected_cost[take] = cost[take]
        selected_factor[take] = factor
        selected_goal[take] = achieved[take]
        selected_achieved_weights[take] = achieved_weights[take]
        valid |= ok
        if step == config.path_projection_steps - 1:
            missing = ~valid
            selected[missing] = candidate[missing]
            selected_metrics[missing] = metrics[missing]
            selected_reasons[missing] = reasons[missing]
            selected_cost[missing] = cost[missing]
            selected_factor[missing] = factor
            selected_goal[missing] = achieved[missing]
            selected_achieved_weights[missing] = achieved_weights[missing]
    return (
        selected,
        valid,
        full_valid,
        selected_reasons,
        selected_metrics,
        selected_cost,
        selected_goal,
        selected_achieved_weights,
    )


def _adaptive_radius(
    state: dict[str, torch.Tensor | float | int], config: TTOConfig, iteration: int
) -> tuple[float, float]:
    if iteration == 0:
        return (1.0, float(state["spread"]))
    if not config.path_adaptive_shrink:
        return (config.endpoint_shrink**iteration, 0.0)
    scores = state["observed_scores"]
    observations = state["observations"]
    center = state["center"] @ state["endpoints"]
    high = scores >= scores.max() - config.path_region_margin
    if int(high.sum()) < 2:
        return (config.path_min_radius, 0.0)
    distance = torch.linalg.vector_norm(observations[high] - center[None], dim=-1)
    extent = float(torch.quantile(distance, 0.9))
    spread = max(float(state["spread"]), 1e-06)
    radius = min(config.path_radius_cap, max(config.path_min_radius, extent / spread))
    return (radius, extent)


@torch.no_grad()
def optimize_path_endpoint_surface(
    proposals: torch.Tensor,
    original_scores: torch.Tensor,
    anchor: torch.Tensor,
    score_fn: PathEndpointScoreFn,
    *,
    config: TTOConfig,
    generators: Sequence[torch.Generator],
    ego_history: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> PathEndpointResult:
    """Search endpoint regions and the conditional paths which reach them."""
    if proposals.ndim != 4 or proposals.shape[-1] != 3 or proposals.shape[-2] < 3:
        raise ValueError("proposals must have shape [B,N,T>=3,3]")
    batch, proposal_count, steps, _ = proposals.shape
    if anchor.shape != (batch, steps, 3) or original_scores.shape != (
        batch,
        proposal_count,
    ):
        raise ValueError("anchor/scores are incompatible with proposals")
    if len(generators) != batch:
        raise ValueError("one per-scene generator is required")
    if (
        ego_history.ndim != 3
        or ego_history.shape[0] != batch
        or ego_history.shape[1] < 3
        or (ego_history.shape[2] < 7)
    ):
        raise ValueError("ego_history must have shape [B,H>=3,F>=7]")
    if any(
        (
            value.device != proposals.device
            for value in (original_scores, anchor, ego_history)
        )
    ):
        raise ValueError("all path endpoint inputs must be on the same device")
    if (
        not torch.isfinite(anchor).all()
        or not torch.isfinite(ego_history[..., :7]).all()
    ):
        raise ValueError("anchor and three-frame history must be finite")
    if output_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("output_dtype must be floating point")
    device = proposals.device
    clean_input = proposals.float()
    scores = original_scores.float()
    finite = torch.isfinite(clean_input).flatten(2).all(-1) & torch.isfinite(scores)
    clean = torch.where(finite[..., None, None], clean_input, anchor.float()[:, None])
    rounds = config.endpoint_rounds
    modes = config.endpoint_modes
    goals_per_mode = config.endpoint_samples_per_mode
    variants = config.path_variants_per_goal
    query_count = modes * goals_per_mode * variants
    shortlist = config.endpoint_shortlist
    fill = anchor.to(output_dtype)
    history_audit = [
        _history_summary(ego_history[row], config.history_dt) for row in range(batch)
    ]
    candidates = fill[:, None].expand(-1, shortlist, -1, -1).clone()
    available = torch.zeros(batch, shortlist, device=device, dtype=torch.bool)
    parents = torch.zeros(batch, shortlist, device=device, dtype=torch.long)
    details: dict[str, torch.Tensor] = {
        "original_proposals": clean_input.clone(),
        "original_scores": scores.clone(),
        "original_finite": finite,
        "history_pose": ego_history[:, -3:, :3].float().clone(),
        "history_velocity": ego_history[:, -1, 3:5].float().clone(),
        "history_acceleration": ego_history[:, -1, 5:7].float().clone(),
        "history_pose_velocity": torch.stack(
            [item["pose_velocity"] for item in history_audit]
        ),
        "history_pose_acceleration": torch.stack(
            [item["pose_acceleration"] for item in history_audit]
        ),
        "history_past_yaw_rate": torch.stack(
            [item["past_yaw_rate"] for item in history_audit]
        ),
        "history_past_yaw_acceleration": torch.stack(
            [item["past_yaw_acceleration"] for item in history_audit]
        ),
        "mode_parents": torch.full((batch, modes), -1, device=device, dtype=torch.long),
        "mode_members": torch.zeros(
            batch, modes, proposal_count, device=device, dtype=torch.bool
        ),
        "mode_active": torch.zeros(batch, modes, device=device, dtype=torch.bool),
        "mode_center_goals": torch.zeros(batch, rounds + 1, modes, 2, device=device),
        "mode_radii": torch.zeros(batch, rounds, modes, device=device),
        "mode_region_extent": torch.zeros(batch, rounds, modes, device=device),
        "trial_goals": torch.zeros(
            batch, rounds, modes, config.endpoint_pool_size, 2, device=device
        ),
        "trial_surface_scores": torch.full(
            (batch, rounds, modes, config.endpoint_pool_size), -torch.inf, device=device
        ),
        "trial_quality_scores": torch.full(
            (batch, rounds, modes, config.endpoint_pool_size), -torch.inf, device=device
        ),
        "trial_reachability": torch.zeros(
            batch, rounds, modes, config.endpoint_pool_size, device=device
        ),
        "query_trajectories": fill[:, None, None]
        .expand(-1, rounds, query_count, -1, -1)
        .clone(),
        "query_requested_goals": torch.zeros(
            batch, rounds, query_count, 2, device=device
        ),
        "query_achieved_goals": torch.zeros(
            batch, rounds, query_count, 2, device=device
        ),
        "query_source_weights": torch.zeros(
            batch, rounds, query_count, proposal_count, device=device
        ),
        "query_achieved_weights": torch.zeros(
            batch, rounds, query_count, proposal_count, device=device
        ),
        "query_parents": torch.full(
            (batch, rounds, query_count), -1, device=device, dtype=torch.long
        ),
        "query_goal_ids": torch.full(
            (batch, rounds, query_count), -1, device=device, dtype=torch.long
        ),
        "query_variant_ids": torch.full(
            (batch, rounds, query_count), -1, device=device, dtype=torch.long
        ),
        "query_projection_factors": torch.zeros(
            batch, rounds, query_count, device=device
        ),
        "query_full_goal_valid": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.bool
        ),
        "query_geometry_valid": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.bool
        ),
        "query_endpoint_novel": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.bool
        ),
        "query_valid": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.bool
        ),
        "query_rejection_bits": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.long
        ),
        "query_geometry_metrics": torch.zeros(
            batch, rounds, query_count, 12, device=device
        ),
        "query_dynamics_cost": torch.zeros(batch, rounds, query_count, device=device),
        "query_proxy_scores": torch.full(
            (batch, rounds, query_count), -torch.inf, device=device
        ),
        "query_effective_scores": torch.full(
            (batch, rounds, query_count), -torch.inf, device=device
        ),
        "query_observed": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.bool
        ),
        "group_values": torch.full(
            (batch, rounds, modes, goals_per_mode), -torch.inf, device=device
        ),
        "group_quality_values": torch.full(
            (batch, rounds, modes, goals_per_mode), -torch.inf, device=device
        ),
        "group_best_scores": torch.full(
            (batch, rounds, modes, goals_per_mode), -torch.inf, device=device
        ),
        "group_mean_scores": torch.full(
            (batch, rounds, modes, goals_per_mode), -torch.inf, device=device
        ),
        "group_feasible_fraction": torch.zeros(
            batch, rounds, modes, goals_per_mode, device=device
        ),
        "group_score_std": torch.zeros(
            batch, rounds, modes, goals_per_mode, device=device
        ),
        "group_best_query": torch.full(
            (batch, rounds, modes, goals_per_mode), -1, device=device, dtype=torch.long
        ),
        "selected_query_indices": torch.full(
            (batch, shortlist), -1, device=device, dtype=torch.long
        ),
        "selected_proxy_scores": torch.full(
            (batch, shortlist), -torch.inf, device=device
        ),
        "selected_effective_scores": torch.full(
            (batch, shortlist), -torch.inf, device=device
        ),
        "scorer_rounds": torch.zeros(batch, device=device, dtype=torch.long),
        "generated_slots": torch.zeros(batch, device=device, dtype=torch.long),
        "completed_slots": torch.zeros(batch, device=device, dtype=torch.long),
        "scored_slots": torch.zeros(batch, device=device, dtype=torch.long),
    }
    states: list[list[dict[str, torch.Tensor | float | int]]] = []
    for b in range(batch):
        scene_states: list[dict[str, torch.Tensor | float | int]] = []
        for mode, indices in enumerate(
            _neighborhoods(clean[b], scores[b], finite[b], config)
        ):
            endpoints = clean[b, indices, -1, :2]
            member_scores = scores[b, indices]
            best = int(member_scores.argmax())
            center = torch.zeros_like(member_scores)
            center[best] = 1.0
            centered = endpoints - endpoints.mean(0)
            covariance = centered.T @ centered / max(1, len(indices))
            spread = float(
                torch.linalg.vector_norm(endpoints - endpoints[best], dim=-1).max()
            )
            active = len(indices) >= variants and spread > 0.0001
            details["mode_parents"][b, mode] = indices[best]
            details["mode_members"][b, mode, indices] = True
            details["mode_active"][b, mode] = active
            details["mode_center_goals"][b, 0, mode] = center @ endpoints
            scene_states.append(
                {
                    "indices": indices,
                    "endpoints": endpoints,
                    "member_scores": member_scores,
                    "center": center,
                    "best_score": member_scores[best].clone(),
                    "covariance": covariance,
                    "observations": endpoints.clone(),
                    "observed_scores": member_scores.clone(),
                    "reach_observations": endpoints.clone(),
                    "reach_values": torch.ones_like(member_scores),
                    "parent": int(indices[best]),
                    "spread": spread,
                }
            )
        states.append(scene_states)
    scorer_rounds = 0
    for iteration in range(rounds):
        weights = torch.zeros(batch, query_count, proposal_count, device=device)
        query_parents = torch.zeros(batch, query_count, device=device, dtype=torch.long)
        requested = torch.zeros(batch, query_count, 2, device=device)
        attempted = torch.zeros(batch, query_count, device=device, dtype=torch.bool)
        goal_novel = torch.zeros(
            batch, modes, goals_per_mode, device=device, dtype=torch.bool
        )
        for b, scene_states in enumerate(states):
            for mode, state in enumerate(scene_states):
                details["mode_center_goals"][b, iteration, mode] = (
                    state["center"] @ state["endpoints"]
                )
                if not bool(details["mode_active"][b, mode]):
                    continue
                radius, extent = _adaptive_radius(state, config, iteration)
                details["mode_radii"][b, iteration, mode] = radius
                details["mode_region_extent"][b, iteration, mode] = extent
                trial_weights = _mixture_trials(
                    state["endpoints"],
                    state["center"],
                    radius,
                    state["member_scores"],
                    config.endpoint_pool_size,
                    config.endpoint_temperature,
                    generators[b],
                )
                goals = trial_weights @ state["endpoints"]
                bandwidth = (
                    state["covariance"] * radius**2
                    + torch.eye(2, device=device) * config.endpoint_bandwidth_floor**2
                )
                quality = _surface(
                    goals, state["observations"], state["observed_scores"], bandwidth
                )
                reachability = _surface(
                    goals, state["reach_observations"], state["reach_values"], bandwidth
                ).clamp(0.0, 1.0)
                heights = quality - config.path_infeasible_penalty * (
                    1.0 - reachability
                )
                details["trial_goals"][b, iteration, mode] = goals
                details["trial_surface_scores"][b, iteration, mode] = heights
                details["trial_quality_scores"][b, iteration, mode] = quality
                details["trial_reachability"][b, iteration, mode] = reachability
                novelty = torch.cdist(goals, state["observations"]).amin(-1) > 0.001
                eligible_heights = heights.masked_fill(~novelty, -torch.inf)
                elite_count = min(
                    config.endpoint_pool_size,
                    max(goals_per_mode, config.endpoint_pool_size // 4),
                )
                elite = torch.topk(eligible_heights, elite_count).indices
                high_region = torch.zeros(
                    config.endpoint_pool_size, device=device, dtype=torch.bool
                )
                high_region[elite] = novelty[elite]
                chosen_goals: list[int] = []
                for goal_id in range(goals_per_mode):
                    if not bool(high_region.any()):
                        break
                    if not chosen_goals:
                        pick = int(
                            heights.masked_fill(~high_region, -torch.inf).argmax()
                        )
                    else:
                        diversity = torch.cdist(goals, goals[chosen_goals]).amin(-1)
                        pick = int(
                            diversity.masked_fill(~high_region, -torch.inf).argmax()
                        )
                    chosen_goals.append(pick)
                    high_region &= (
                        torch.linalg.vector_norm(goals - goals[pick], dim=-1) > 0.001
                    )
                    goal_novel[b, mode, goal_id] = True
                    local_weights = trial_weights[pick]
                    selected_parents = _parent_variants(
                        clean[b],
                        state["indices"],
                        local_weights,
                        variants,
                        config.path_min_variant_ade,
                    )
                    for variant_id, parent in enumerate(selected_parents):
                        query = (
                            mode * goals_per_mode + goal_id
                        ) * variants + variant_id
                        weights[b, query, state["indices"]] = local_weights
                        query_parents[b, query] = parent
                        requested[b, query] = goals[pick]
                        attempted[b, query] = True
                        details["query_goal_ids"][b, iteration, query] = goal_id
                        details["query_variant_ids"][b, iteration, query] = variant_id
        population = fill[:, None].expand(-1, query_count, -1, -1).clone()
        valid_queries = torch.zeros_like(attempted)
        for b in range(batch):
            slots = attempted[b].nonzero(as_tuple=False).flatten()
            if not len(slots):
                continue
            result = _complete_path_variants(
                clean[b],
                weights[b, slots],
                query_parents[b, slots],
                requested[b, slots],
                config,
                output_dtype,
                ego_history[b],
            )
            (
                trajectories,
                valid,
                full_valid,
                reasons,
                metrics,
                dynamic_cost,
                achieved,
                achieved_weights,
            ) = result
            population[b, slots] = trajectories
            valid_queries[b, slots] = valid
            details["query_requested_goals"][b, iteration, slots] = requested[b, slots]
            details["query_achieved_goals"][b, iteration, slots] = achieved
            details["query_source_weights"][b, iteration, slots] = weights[b, slots]
            details["query_achieved_weights"][b, iteration, slots] = achieved_weights
            details["query_parents"][b, iteration, slots] = query_parents[b, slots]
            details["query_full_goal_valid"][b, iteration, slots] = full_valid
            details["query_geometry_valid"][b, iteration, slots] = valid
            details["query_rejection_bits"][b, iteration, slots] = reasons
            details["query_geometry_metrics"][b, iteration, slots] = metrics
            details["query_dynamics_cost"][b, iteration, slots] = dynamic_cost
            parent_goals = clean[b, query_parents[b, slots], -1, :2]
            denominator = torch.linalg.vector_norm(
                requested[b, slots] - parent_goals, dim=-1
            )
            numerator = torch.linalg.vector_norm(achieved - parent_goals, dim=-1)
            details["query_projection_factors"][b, iteration, slots] = torch.where(
                denominator > 1e-07,
                numerator / denominator,
                torch.ones_like(denominator),
            )
            for mode in range(modes):
                previous: list[torch.Tensor] = []
                for goal_id in range(goals_per_mode):
                    group_start = (mode * goals_per_mode + goal_id) * variants
                    group_slots = torch.arange(
                        group_start, group_start + variants, device=device
                    )
                    group_slots = group_slots[attempted[b, group_slots]]
                    if not len(group_slots):
                        continue
                    cast_goal = requested[b, group_slots[0]].to(output_dtype).float()
                    novel = all(
                        (
                            float(torch.linalg.vector_norm(cast_goal - old)) > 0.001
                            for old in previous
                        )
                    )
                    details["query_endpoint_novel"][b, iteration, group_slots] = novel
                    if not novel:
                        valid_queries[b, group_slots] = False
                    previous.append(cast_goal)
        details["query_trajectories"][:, iteration] = population
        details["query_valid"][:, iteration] = valid_queries
        details["generated_slots"] += attempted.sum(1)
        details["completed_slots"] += valid_queries.sum(1)
        if not bool(valid_queries.any()):
            continue
        scored_population = torch.where(
            valid_queries[..., None, None], population, fill[:, None]
        )
        observed = score_fn(scored_population, valid_queries)
        if not torch.is_tensor(observed) or observed.shape != (batch, query_count):
            raise ValueError("score_fn must return [B,path-aware query count]")
        if observed.device != device:
            raise ValueError("score_fn returned scores on a different device")
        scorer_rounds += 1
        measured = valid_queries & torch.isfinite(observed)
        details["scored_slots"] += measured.sum(1)
        observed = observed.float()
        details["query_observed"][:, iteration] = measured
        details["query_proxy_scores"][:, iteration] = observed.masked_fill(
            ~measured, -torch.inf
        )
        consensus = details["query_geometry_metrics"][:, iteration, :, 9]
        effective = (
            observed
            - config.path_dynamics_weight * details["query_dynamics_cost"][:, iteration]
            - config.path_consensus_weight * consensus
        ).masked_fill(~measured, -torch.inf)
        details["query_effective_scores"][:, iteration] = effective
        for b, scene_states in enumerate(states):
            for mode, state in enumerate(scene_states):
                for goal_id in range(goals_per_mode):
                    start = (mode * goals_per_mode + goal_id) * variants
                    slots = torch.arange(start, start + variants, device=device)
                    attempted_count = int(attempted[b, slots].sum())
                    if attempted_count == 0:
                        continue
                    full_fraction = (
                        float(
                            details["query_full_goal_valid"][b, iteration, slots].sum()
                        )
                        / attempted_count
                    )
                    details["group_feasible_fraction"][
                        b, iteration, mode, goal_id
                    ] = full_fraction
                    request_slot = slots[attempted[b, slots]][0]
                    state["reach_observations"] = torch.cat(
                        (state["reach_observations"], requested[b, request_slot][None])
                    )
                    state["reach_values"] = torch.cat(
                        (state["reach_values"], scores.new_tensor([full_fraction]))
                    )
                    measured_slots = slots[measured[b, slots]]
                    if not len(measured_slots):
                        continue
                    raw_group = observed[b, measured_slots]
                    details["group_best_scores"][
                        b, iteration, mode, goal_id
                    ] = raw_group.max()
                    details["group_mean_scores"][
                        b, iteration, mode, goal_id
                    ] = raw_group.mean()
                    score_std = (
                        raw_group.std(unbiased=False)
                        if len(measured_slots) > 1
                        else raw_group.new_zeros(())
                    )
                    merit_group = effective[b, measured_slots]
                    best_local = int(merit_group.argmax())
                    best_slot = measured_slots[best_local]
                    group_quality = (
                        merit_group[best_local]
                        - config.path_disagreement_weight * score_std
                    )
                    group_value = group_quality - config.path_infeasible_penalty * (
                        1.0 - full_fraction
                    )
                    details["group_score_std"][b, iteration, mode, goal_id] = score_std
                    details["group_values"][b, iteration, mode, goal_id] = group_value
                    details["group_quality_values"][
                        b, iteration, mode, goal_id
                    ] = group_quality
                    details["group_best_query"][b, iteration, mode, goal_id] = best_slot
                    observed_goal = details["query_achieved_goals"][
                        b, iteration, best_slot
                    ]
                    state["observations"] = torch.cat(
                        (state["observations"], observed_goal[None])
                    )
                    state["observed_scores"] = torch.cat(
                        (state["observed_scores"], group_quality.reshape(1))
                    )
                    if (
                        float(
                            torch.linalg.vector_norm(
                                observed_goal - requested[b, request_slot]
                            )
                        )
                        > 0.001
                    ):
                        state["reach_observations"] = torch.cat(
                            (state["reach_observations"], observed_goal[None])
                        )
                        state["reach_values"] = torch.cat(
                            (state["reach_values"], scores.new_ones(1))
                        )
                    if bool(group_value > state["best_score"]):
                        state["best_score"] = group_value.clone()
                        local_achieved = details["query_achieved_weights"][
                            b, iteration, best_slot, state["indices"]
                        ]
                        state["center"] = (
                            local_achieved / local_achieved.sum().clamp_min(1e-08)
                        )
                details["mode_center_goals"][b, iteration + 1, mode] = (
                    state["center"] @ state["endpoints"]
                )
    details["scorer_rounds"].fill_(scorer_rounds)
    for b in range(batch):
        flat_raw = details["query_proxy_scores"][b].flatten()
        flat_merit = details["query_effective_scores"][b].flatten()
        flat_trajectories = details["query_trajectories"][b].flatten(0, 1)
        finite_queries = torch.isfinite(flat_merit)
        reserved: list[int] = []
        per_round = query_count
        for mode in range(modes):
            indices = torch.arange(rounds * per_round, device=device)
            local = indices % per_round // (goals_per_mode * variants) == mode
            masked = flat_merit.masked_fill(~(local & finite_queries), -torch.inf)
            if bool(torch.isfinite(masked).any()):
                reserved.append(int(masked.argmax()))
        reserved.sort(key=lambda index: float(flat_merit[index]), reverse=True)
        ordered = (
            reserved + torch.argsort(flat_merit, descending=True, stable=True).tolist()
        )
        chosen: list[int] = []
        for index in ordered:
            if index in chosen or not bool(finite_queries[index]):
                continue
            if chosen:
                ade = torch.linalg.vector_norm(
                    flat_trajectories[chosen, :, :2].float()
                    - flat_trajectories[index, :, :2].float(),
                    dim=-1,
                ).mean(-1)
                if bool((ade <= 0.0001).any()):
                    continue
            chosen.append(index)
            if len(chosen) == shortlist:
                break
        if chosen:
            length = len(chosen)
            candidates[b, :length] = flat_trajectories[chosen]
            available[b, :length] = True
            flat_parents = details["query_parents"][b].flatten()
            parents[b, :length] = flat_parents[chosen]
            details["selected_query_indices"][b, :length] = torch.tensor(
                chosen, device=device
            )
            details["selected_proxy_scores"][b, :length] = flat_raw[chosen]
            details["selected_effective_scores"][b, :length] = flat_merit[chosen]
    return PathEndpointResult(candidates, available, parents, details)
