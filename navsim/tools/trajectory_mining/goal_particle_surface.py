"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
from typing import Sequence, Callable
import torch
from navsim.tools.trajectory_mining.config import TTOConfig
from navsim.tools.trajectory_mining.path_endpoint_surface import _complete_path_variants, _parent_variants
from navsim.tools.trajectory_mining.endpoint_surface import _wrap
from dataclasses import dataclass
import math

GoalParticleScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GoalParticleResult:
    """A bounded shortlist plus explicit goal/path provenance."""

    candidates: torch.Tensor
    available: torch.Tensor
    parents: torch.Tensor
    requested_states: torch.Tensor
    achieved_states: torch.Tensor
    raw_scores: torch.Tensor
    effective_scores: torch.Tensor
    details: dict[str, torch.Tensor]


def _terminal_states(trajectories: torch.Tensor, dt: float) -> torch.Tensor:
    """Return ``[..., x, y, heading, terminal_speed]``."""
    if trajectories.shape[-2] < 2 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories must end in [T>=2,3]")
    speed = (
        torch.linalg.vector_norm(
            trajectories[..., -1, :2].float() - trajectories[..., -2, :2].float(),
            dim=-1,
        )
        / dt
    )
    return torch.cat(
        (
            trajectories[..., -1, :2].float(),
            _wrap(trajectories[..., -1, 2].float())[..., None],
            speed[..., None],
        ),
        dim=-1,
    )


def _xy_bandwidth(states: torch.Tensor, floor: float) -> float:
    """Robust local scale; never use the extent of the entire endpoint cloud."""
    if states.shape[0] < 2:
        return floor
    distance = torch.cdist(states[:, :2], states[:, :2])
    distance.fill_diagonal_(torch.inf)
    nearest = distance.amin(-1)
    nearest = nearest[torch.isfinite(nearest) & (nearest > 1e-06)]
    if not len(nearest):
        return floor
    return max(floor, float(nearest.median()))


def _state_distance_squared(
    queries: torch.Tensor,
    observations: torch.Tensor,
    config: TTOConfig,
    xy_scale: float,
) -> torch.Tensor:
    xy = (
        torch.linalg.vector_norm(
            queries[:, None, :2] - observations[None, :, :2], dim=-1
        )
        / xy_scale
    )
    heading = (
        _wrap(queries[:, None, 2] - observations[None, :, 2]).abs()
        / config.goal_particle_heading_scale
    )
    speed = (
        queries[:, None, 3] - observations[None, :, 3]
    ).abs() / config.goal_particle_speed_scale
    return xy.square() + heading.square() + speed.square()


def _knn_surface(
    queries: torch.Tensor,
    observations: torch.Tensor,
    values: torch.Tensor,
    config: TTOConfig,
    xy_scale: float,
) -> torch.Tensor:
    """Local kernel regression over discrete measured particles."""
    distance = _state_distance_squared(queries, observations, config, xy_scale)
    count = min(config.goal_particle_neighbors, observations.shape[0])
    nearest_distance, nearest = torch.topk(distance, count, dim=-1, largest=False)
    kernel = torch.softmax(-0.5 * nearest_distance, dim=-1)
    return (kernel * values[nearest]).sum(-1)


def _particle_trials(
    states: torch.Tensor,
    values: torch.Tensor,
    support_weights: torch.Tensor,
    reach_states: torch.Tensor,
    reach_values: torch.Tensor,
    iteration: int,
    config: TTOConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample local convex particles from one global archive.

    Returns requested terminal states, their original-proposal support weights,
    source particle indices, predicted quality, and predicted reachability.
    """
    pool = config.goal_particle_pool_size
    xy_scale = _xy_bandwidth(states, config.goal_particle_min_xy_bandwidth)
    probability = torch.softmax(
        (values - values.max()) / config.goal_particle_temperature, dim=0
    )
    bases = torch.multinomial(probability, pool, replacement=True, generator=generator)
    bases[0] = values.argmax()
    pairwise = _state_distance_squared(states, states, config, xy_scale)
    neighbor_count = min(config.goal_particle_neighbors, states.shape[0])
    neighborhoods = torch.topk(pairwise, neighbor_count, dim=-1, largest=False).indices
    radius = min(
        1.0,
        config.goal_particle_jitter_scale
        * config.goal_particle_region_shrink**iteration,
    )
    local_neighbors = neighborhoods[bases]
    take = min(3, neighbor_count)
    random_order = torch.rand(
        pool, neighbor_count, generator=generator, device=states.device
    ).argsort(dim=-1, stable=True)
    chosen = torch.gather(local_neighbors, 1, random_order[:, :take])
    missing_base = ~(chosen == bases[:, None]).any(-1)
    chosen[missing_base, -1] = bases[missing_base]
    random_mass = -torch.log(
        torch.rand(pool, take, generator=generator, device=states.device).clamp_min(
            1e-07
        )
    )
    local_values = values[chosen]
    score_prior = torch.softmax(
        (
            (local_values - local_values.amax(-1, keepdim=True))
            / config.goal_particle_temperature
        ).clamp_min(-20.0),
        dim=-1,
    )
    local_mass = random_mass * score_prior.clamp_min(0.0001)
    local_mass /= local_mass.sum(-1, keepdim=True).clamp_min(1e-08)
    mixtures = states.new_zeros(pool, states.shape[0])
    mixtures.scatter_add_(1, bases[:, None], mixtures.new_full((pool, 1), 1.0 - radius))
    mixtures.scatter_add_(1, chosen, radius * local_mass)
    requested_xy = mixtures @ states[:, :2]
    base_xy = states[bases, :2]
    displacement = requested_xy - base_xy
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    cap = (config.goal_particle_max_xy_shift / distance.clamp_min(1e-08)).clamp(max=1.0)
    capped = mixtures.new_zeros(mixtures.shape)
    capped[torch.arange(pool, device=states.device), bases] = 1.0
    mixtures = capped + cap[:, None] * (mixtures - capped)
    requested = states.new_zeros(pool, 4)
    requested[:, :2] = mixtures @ states[:, :2]
    requested[:, 2] = torch.atan2(
        mixtures @ torch.sin(states[:, 2]), mixtures @ torch.cos(states[:, 2])
    )
    requested[:, 3] = mixtures @ states[:, 3]
    original_support = mixtures @ support_weights
    quality = _knn_surface(requested, states, values, config, xy_scale)
    reach_scale = _xy_bandwidth(reach_states, config.goal_particle_min_xy_bandwidth)
    reachability = _knn_surface(
        requested, reach_states, reach_values, config, reach_scale
    ).clamp(0.0, 1.0)
    return (requested, original_support, bases, quality, reachability)


def _select_high_region(
    states: torch.Tensor,
    merit: torch.Tensor,
    observations: torch.Tensor,
    count: int,
    min_separation: float,
    elite_fraction: float,
) -> list[int]:
    """Select only from the global high-value set, without peak partitioning."""
    novelty = torch.cdist(states[:, :2], observations[:, :2]).amin(-1) > 0.001
    eligible_merit = merit.masked_fill(~novelty, -torch.inf)
    elite_count = min(len(states), max(count, math.ceil(len(states) * elite_fraction)))
    elite = torch.topk(eligible_merit, elite_count).indices
    high = torch.zeros(len(states), device=states.device, dtype=torch.bool)
    high[elite] = novelty[elite]
    selected: list[int] = []
    order_tensor = torch.argsort(merit, descending=True, stable=True)
    order = order_tensor.detach().cpu().tolist()
    ordered_high = high[order_tensor].detach().cpu().tolist()
    xy = states[:, :2].detach().cpu()
    for index, is_high in zip(order, ordered_high):
        if not is_high:
            continue
        if selected:
            separation = torch.linalg.vector_norm(xy[index] - xy[selected], dim=-1)
            if bool((separation < min_separation).any()):
                continue
        selected.append(index)
        if len(selected) == count:
            break
    if len(selected) < count:
        for index, is_high in zip(order, ordered_high):
            if is_high and index not in selected:
                selected.append(index)
            if len(selected) == count:
                break
    return selected


def _pseudo_shadow_metrics(
    trajectories: torch.Tensor, parents: torch.Tensor, dt: float, config: TTOConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Proposal-support proxy; it is not a map or actor collision check."""
    candidate = trajectories.float()
    parent = parents.float()
    candidate_state = _terminal_states(candidate, dt)
    parent_state = _terminal_states(parent, dt)
    metrics = torch.stack(
        (
            torch.linalg.vector_norm(candidate[..., :2] - parent[..., :2], dim=-1).mean(
                -1
            ),
            torch.linalg.vector_norm(
                candidate[..., -1, :2] - parent[..., -1, :2], dim=-1
            ),
            _wrap(candidate[..., 2] - parent[..., 2]).abs().amax(-1),
            (candidate_state[..., 3] - parent_state[..., 3]).abs(),
        ),
        dim=-1,
    )
    limits = metrics.new_tensor(
        (
            config.goal_particle_shadow_max_ade,
            config.goal_particle_shadow_max_endpoint,
            config.goal_particle_shadow_max_heading,
            config.goal_particle_shadow_max_speed,
        )
    )
    bad = metrics > limits[None]
    finite = torch.isfinite(metrics).all(-1)
    weights = 2 ** torch.arange(4, device=metrics.device, dtype=torch.long)
    reasons = (bad.long() * weights).sum(-1) + (~finite).long() * 16
    cost = (metrics / limits[None]).square().mean(-1)
    return (finite & ~bad.any(-1), reasons, metrics, cost)


def _reach_cost(
    requested: torch.Tensor, achieved: torch.Tensor, config: TTOConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    components = torch.stack(
        (
            torch.linalg.vector_norm(achieved[:, :2] - requested[:, :2], dim=-1),
            _wrap(achieved[:, 2] - requested[:, 2]).abs(),
            (achieved[:, 3] - requested[:, 3]).abs(),
        ),
        dim=-1,
    )
    scales = components.new_tensor(
        (
            config.goal_particle_max_xy_shift,
            config.goal_particle_heading_scale,
            config.goal_particle_speed_scale,
        )
    )
    return (components, (components / scales[None]).square().mean(-1))


@torch.no_grad()
def optimize_goal_particle_surface(
    proposals: torch.Tensor,
    original_scores: torch.Tensor,
    anchor: torch.Tensor,
    score_fn: GoalParticleScoreFn,
    *,
    config: TTOConfig,
    generators: Sequence[torch.Generator],
    ego_history: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> GoalParticleResult:
    """Run bounded global particle search and return complete paths only."""
    if proposals.ndim != 4 or proposals.shape[-1] != 3 or proposals.shape[-2] < 3:
        raise ValueError("proposals must have shape [B,N,T>=3,3]")
    batch, proposal_count, steps, _ = proposals.shape
    if original_scores.shape != (batch, proposal_count):
        raise ValueError("original_scores must have shape [B,N]")
    if anchor.shape != (batch, steps, 3):
        raise ValueError("anchor must have shape [B,T,3]")
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
        raise ValueError("all goal-particle inputs must be on the same device")
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
    original = proposals.float()
    scores = original_scores.float()
    finite = torch.isfinite(original).flatten(2).all(-1) & torch.isfinite(scores)
    clean = torch.where(finite[..., None, None], original, anchor.float()[:, None])
    rounds = config.goal_particle_rounds
    goals_per_round = config.goal_particle_goals_per_round
    variants = config.path_variants_per_goal
    query_count = goals_per_round * variants
    pool = config.goal_particle_pool_size
    shortlist = config.goal_particle_shortlist
    maximum_particles = proposal_count + rounds * query_count
    fill = anchor.to(output_dtype)
    anchor_states = _terminal_states(anchor, config.trajectory_dt)
    candidates = fill[:, None].expand(-1, shortlist, -1, -1).clone()
    available = torch.zeros(batch, shortlist, device=device, dtype=torch.bool)
    parents = torch.zeros(batch, shortlist, device=device, dtype=torch.long)
    requested_output = anchor_states[:, None].expand(-1, shortlist, -1).clone()
    achieved_output = requested_output.clone()
    raw_output = torch.full((batch, shortlist), -torch.inf, device=device)
    effective_output = raw_output.clone()
    details: dict[str, torch.Tensor] = {
        "original_proposals": original.clone(),
        "original_scores": scores.clone(),
        "original_finite": finite,
        "original_states": _terminal_states(clean, config.trajectory_dt),
        "particle_states": torch.zeros(
            batch, rounds + 1, maximum_particles, 4, device=device
        ),
        "particle_scores": torch.full(
            (batch, rounds + 1, maximum_particles), -torch.inf, device=device
        ),
        "particle_available": torch.zeros(
            batch, rounds + 1, maximum_particles, device=device, dtype=torch.bool
        ),
        "particle_counts": torch.zeros(
            batch, rounds + 1, device=device, dtype=torch.long
        ),
        "region_scales": torch.tensor(
            [
                config.goal_particle_jitter_scale
                * config.goal_particle_region_shrink**iteration
                for iteration in range(rounds)
            ],
            device=device,
        )
        .expand(batch, -1)
        .clone(),
        "pool_requested_states": torch.zeros(batch, rounds, pool, 4, device=device),
        "pool_base_particles": torch.full(
            (batch, rounds, pool), -1, device=device, dtype=torch.long
        ),
        "pool_quality": torch.full((batch, rounds, pool), -torch.inf, device=device),
        "pool_reachability": torch.zeros(batch, rounds, pool, device=device),
        "pool_merit": torch.full((batch, rounds, pool), -torch.inf, device=device),
        "selected_pool_indices": torch.full(
            (batch, rounds, goals_per_round), -1, device=device, dtype=torch.long
        ),
        "query_trajectories": fill[:, None, None]
        .expand(-1, rounds, query_count, -1, -1)
        .clone(),
        "query_requested_states": torch.zeros(
            batch, rounds, query_count, 4, device=device
        ),
        "query_achieved_states": torch.zeros(
            batch, rounds, query_count, 4, device=device
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
        "query_attempted": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.bool
        ),
        "query_geometry_valid": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.bool
        ),
        "query_shadow_valid": torch.zeros(
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
        "query_shadow_metrics": torch.zeros(
            batch, rounds, query_count, 4, device=device
        ),
        "query_dynamics_cost": torch.zeros(batch, rounds, query_count, device=device),
        "query_reach_components": torch.zeros(
            batch, rounds, query_count, 3, device=device
        ),
        "query_reach_cost": torch.zeros(batch, rounds, query_count, device=device),
        "query_shadow_cost": torch.zeros(batch, rounds, query_count, device=device),
        "query_raw_scores": torch.full(
            (batch, rounds, query_count), -torch.inf, device=device
        ),
        "query_effective_scores": torch.full(
            (batch, rounds, query_count), -torch.inf, device=device
        ),
        "query_observed": torch.zeros(
            batch, rounds, query_count, device=device, dtype=torch.bool
        ),
        "goal_feasible_fraction": torch.zeros(
            batch, rounds, goals_per_round, device=device
        ),
        "selected_query_indices": torch.full(
            (batch, shortlist), -1, device=device, dtype=torch.long
        ),
        "generated_slots": torch.zeros(batch, device=device, dtype=torch.long),
        "completed_slots": torch.zeros(batch, device=device, dtype=torch.long),
        "scored_slots": torch.zeros(batch, device=device, dtype=torch.long),
        "scorer_rounds": torch.zeros(batch, device=device, dtype=torch.long),
    }
    archives: list[dict[str, torch.Tensor]] = []
    eye = torch.eye(proposal_count, device=device)
    initial_states = details["original_states"]
    for row in range(batch):
        keep = finite[row]
        archives.append(
            {
                "states": initial_states[row, keep].clone(),
                "values": scores[row, keep].clone(),
                "support": eye[keep].clone(),
                "reach_states": initial_states[row, keep].clone(),
                "reach_values": torch.ones(int(keep.sum()), device=device),
            }
        )

    def snapshot(round_index: int) -> None:
        for row, archive in enumerate(archives):
            count = min(maximum_particles, archive["states"].shape[0])
            details["particle_states"][row, round_index, :count] = archive["states"][
                :count
            ]
            details["particle_scores"][row, round_index, :count] = archive["values"][
                :count
            ]
            details["particle_available"][row, round_index, :count] = True
            details["particle_counts"][row, round_index] = count

    snapshot(0)
    scorer_rounds = 0
    all_indices = torch.arange(proposal_count, device=device)
    for iteration in range(rounds):
        population = fill[:, None].expand(-1, query_count, -1, -1).clone()
        requested = torch.zeros(batch, query_count, 4, device=device)
        source_weights = torch.zeros(batch, query_count, proposal_count, device=device)
        query_parents = torch.zeros(batch, query_count, device=device, dtype=torch.long)
        attempted = torch.zeros(batch, query_count, device=device, dtype=torch.bool)
        for row, archive in enumerate(archives):
            if archive["states"].shape[0] < 2:
                continue
            trial = _particle_trials(
                archive["states"],
                archive["values"],
                archive["support"],
                archive["reach_states"],
                archive["reach_values"],
                iteration,
                config,
                generators[row],
            )
            trial_states, trial_support, bases, quality, reachability = trial
            merit = quality - config.goal_particle_reach_weight * (1.0 - reachability)
            details["pool_requested_states"][row, iteration] = trial_states
            details["pool_base_particles"][row, iteration] = bases
            details["pool_quality"][row, iteration] = quality
            details["pool_reachability"][row, iteration] = reachability
            details["pool_merit"][row, iteration] = merit
            selected = _select_high_region(
                trial_states,
                merit,
                archive["states"],
                goals_per_round,
                config.goal_particle_min_separation,
                config.goal_particle_elite_fraction,
            )
            for goal_id, pool_index in enumerate(selected):
                details["selected_pool_indices"][row, iteration, goal_id] = pool_index
                local_weights = trial_support[pool_index]
                selected_parents = _parent_variants(
                    clean[row],
                    all_indices,
                    local_weights,
                    variants,
                    config.path_min_variant_ade,
                )
                for variant_id, parent in enumerate(selected_parents):
                    slot = goal_id * variants + variant_id
                    attempted[row, slot] = True
                    requested[row, slot] = trial_states[pool_index]
                    source_weights[row, slot] = local_weights
                    query_parents[row, slot] = parent
                    details["query_goal_ids"][row, iteration, slot] = goal_id
                    details["query_variant_ids"][row, iteration, slot] = variant_id
        geometry_valid = torch.zeros_like(attempted)
        shadow_valid_queries = torch.zeros_like(attempted)
        final_valid = torch.zeros_like(attempted)
        for row in range(batch):
            slots = attempted[row].nonzero(as_tuple=False).flatten()
            if not len(slots):
                continue
            completed = _complete_path_variants(
                clean[row],
                source_weights[row, slots],
                query_parents[row, slots],
                requested[row, slots, :2],
                config,
                output_dtype,
                ego_history[row],
            )
            (
                trajectory,
                valid,
                _full_valid,
                reasons,
                metrics,
                dynamics,
                _achieved_xy,
                achieved_weights,
            ) = completed
            population[row, slots] = trajectory
            achieved = _terminal_states(trajectory, config.trajectory_dt)
            shadow_valid, shadow_reasons, shadow_metrics, shadow_cost = (
                _pseudo_shadow_metrics(
                    trajectory,
                    clean[row, query_parents[row, slots]],
                    config.trajectory_dt,
                    config,
                )
            )
            reach_components, reach_cost = _reach_cost(
                requested[row, slots], achieved, config
            )
            accepted = valid & shadow_valid & torch.isfinite(achieved).all(-1)
            geometry_valid[row, slots] = valid
            shadow_valid_queries[row, slots] = shadow_valid
            final_valid[row, slots] = accepted
            details["query_achieved_states"][row, iteration, slots] = achieved
            details["query_achieved_weights"][row, iteration, slots] = achieved_weights
            details["query_geometry_metrics"][row, iteration, slots] = metrics
            details["query_shadow_metrics"][row, iteration, slots] = shadow_metrics
            details["query_dynamics_cost"][row, iteration, slots] = dynamics
            details["query_reach_components"][row, iteration, slots] = reach_components
            details["query_reach_cost"][row, iteration, slots] = reach_cost
            details["query_shadow_cost"][row, iteration, slots] = shadow_cost
            details["query_rejection_bits"][row, iteration, slots] = reasons + (
                shadow_reasons << 10
            )
            for goal_id in range(goals_per_round):
                goal_slots = torch.arange(
                    goal_id * variants, (goal_id + 1) * variants, device=device
                )
                used = goal_slots[attempted[row, goal_slots]]
                if not len(used):
                    continue
                fraction = final_valid[row, used].float().mean()
                details["goal_feasible_fraction"][row, iteration, goal_id] = fraction
                archive = archives[row]
                archive["reach_states"] = torch.cat(
                    (archive["reach_states"], requested[row, used[0]][None])
                )
                archive["reach_values"] = torch.cat(
                    (archive["reach_values"], fraction.reshape(1))
                )
        details["query_trajectories"][:, iteration] = population
        details["query_requested_states"][:, iteration] = requested
        details["query_source_weights"][:, iteration] = source_weights
        details["query_parents"][:, iteration] = torch.where(
            attempted, query_parents, torch.full_like(query_parents, -1)
        )
        details["query_attempted"][:, iteration] = attempted
        details["query_geometry_valid"][:, iteration] = geometry_valid
        details["query_shadow_valid"][:, iteration] = shadow_valid_queries
        details["query_valid"][:, iteration] = final_valid
        details["generated_slots"] += attempted.sum(-1)
        details["completed_slots"] += final_valid.sum(-1)
        if bool(final_valid.any()):
            scored_population = torch.where(
                final_valid[..., None, None], population, fill[:, None]
            )
            observed = score_fn(scored_population, final_valid)
            if not torch.is_tensor(observed) or observed.shape != (batch, query_count):
                raise ValueError("score_fn must return [B,goal-particle query count]")
            if observed.device != device:
                raise ValueError("score_fn returned scores on a different device")
            scorer_rounds += 1
            measured = final_valid & torch.isfinite(observed)
            raw = observed.float().masked_fill(~measured, -torch.inf)
            effective = (
                observed.float()
                - config.goal_particle_dynamics_weight
                * details["query_dynamics_cost"][:, iteration]
                - config.goal_particle_reach_weight
                * details["query_reach_cost"][:, iteration]
                - config.goal_particle_shadow_weight
                * details["query_shadow_cost"][:, iteration]
            ).masked_fill(~measured, -torch.inf)
            details["query_observed"][:, iteration] = measured
            details["query_raw_scores"][:, iteration] = raw
            details["query_effective_scores"][:, iteration] = effective
            details["scored_slots"] += measured.sum(-1)
            for row, archive in enumerate(archives):
                slots = measured[row].nonzero(as_tuple=False).flatten()
                if not len(slots):
                    continue
                archive["states"] = torch.cat(
                    (
                        archive["states"],
                        details["query_achieved_states"][row, iteration, slots],
                    )
                )
                archive["values"] = torch.cat(
                    (archive["values"], effective[row, slots])
                )
                archive["support"] = torch.cat(
                    (
                        archive["support"],
                        details["query_achieved_weights"][row, iteration, slots],
                    )
                )
                archive["reach_states"] = torch.cat(
                    (
                        archive["reach_states"],
                        details["query_achieved_states"][row, iteration, slots],
                    )
                )
                archive["reach_values"] = torch.cat(
                    (archive["reach_values"], torch.ones(len(slots), device=device))
                )
        snapshot(iteration + 1)
    details["scorer_rounds"].fill_(scorer_rounds)
    for row in range(batch):
        flat_effective = details["query_effective_scores"][row].flatten()
        flat_raw = details["query_raw_scores"][row].flatten()
        flat_trajectories = details["query_trajectories"][row].flatten(0, 1)
        flat_parents = details["query_parents"][row].flatten()
        flat_requested = details["query_requested_states"][row].flatten(0, 1)
        flat_achieved = details["query_achieved_states"][row].flatten(0, 1)
        chosen: list[int] = []
        for index in torch.argsort(
            flat_effective, descending=True, stable=True
        ).tolist():
            if not bool(torch.isfinite(flat_effective[index])):
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
            candidates[row, :length] = flat_trajectories[chosen]
            available[row, :length] = True
            parents[row, :length] = flat_parents[chosen]
            requested_output[row, :length] = flat_requested[chosen]
            achieved_output[row, :length] = flat_achieved[chosen]
            raw_output[row, :length] = flat_raw[chosen]
            effective_output[row, :length] = flat_effective[chosen]
            details["selected_query_indices"][row, :length] = torch.tensor(
                chosen, device=device
            )
    return GoalParticleResult(
        candidates=candidates,
        available=available,
        parents=parents,
        requested_states=requested_output,
        achieved_states=achieved_output,
        raw_scores=raw_output,
        effective_scores=effective_output,
        details=details,
    )
