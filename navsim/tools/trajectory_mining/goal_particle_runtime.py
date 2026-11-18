"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
import torch
from navsim.tools.trajectory_mining.cem import make_sample_generators
from navsim.tools.trajectory_mining.endpoint_runtime import _verify_shortlist, _checked_scores
from navsim.tools.trajectory_mining.path_endpoint_surface import _history_summary, _path_metrics


def _prepare_inputs(optimizer, predictions, features, sample_ids):
    """Validate and reproduce the exact frozen Stage3 proposal context."""
    cfg = optimizer.config
    context = optimizer._captured_context()
    original = predictions.get(cfg.trajectory_key)
    if not torch.is_tensor(original) or original.ndim != 3 or original.shape[-1] != 3:
        raise ValueError("goal_particle_rediffusion requires trajectory [B,T,3]")
    if context.module.training:
        raise RuntimeError("goal-particle scorer must be in eval mode")
    if optimizer._postprocessor_paths and (not cfg.allow_postprocessor_bypass):
        raise RuntimeError(
            "goal_particle_rediffusion would bypass the agent's post-scorer safety gate"
        )
    captured = context.kwargs.get("proposals")
    if captured is None and context.args:
        captured = context.args[0]
    if not torch.is_tensor(captured):
        raise ValueError(
            "goal-particle search requires the captured Stage3 proposal pool"
        )
    batch, poses, _ = original.shape
    if (
        batch < 1
        or captured.ndim != 4
        or captured.shape[0] != batch
        or (captured.shape[1] < 2)
        or (captured.shape[2:] != original.shape[1:])
        or (
            captured.shape[1]
            != int(getattr(context.module, "proposal_num", captured.shape[1]))
        )
        or (poses != int(getattr(context.module, "poses_num", poses)))
    ):
        raise ValueError(
            "captured goal-particle pool is incompatible with scorer horizon"
        )
    if sample_ids is not None and len(sample_ids) != batch:
        raise ValueError("sample_ids length must equal batch")
    reported = predictions.get(cfg.proposals_key)
    if reported is not None:
        if (
            not torch.is_tensor(reported)
            or reported.shape != captured.shape
            or (not torch.equal(reported.detach().to(captured), captured.detach()))
        ):
            raise RuntimeError(
                "reported proposals differ from the captured scorer pool"
            )
    source = reported if reported is not None else captured
    executable = source.detach().to(device=original.device, dtype=original.dtype)
    if not torch.isfinite(executable).all() or not torch.isfinite(original).all():
        raise FloatingPointError("non-finite Stage3 trajectory/proposals")
    errors = (
        (executable.float() - original.detach().float()[:, None])
        .abs()
        .flatten(2)
        .amax(-1)
    )
    slots = errors.argmin(1)
    rows = torch.arange(batch, device=original.device)
    unmatched = errors[rows, slots] > 1e-05
    if bool(unmatched.any()):
        exact = (
            (captured.detach() == original.detach().to(captured)[:, None])
            .flatten(2)
            .all(-1)
        )
        if not bool(exact[unmatched].any(1).all()):
            raise RuntimeError(
                "Stage3 output must belong to its captured proposal pool"
            )
        slots = torch.where(unmatched, exact.long().argmax(1).to(slots.device), slots)
    reference = source.detach().to(device=original.device).clone()
    reference[rows, slots] = original.detach().to(reference)
    expected = predictions.get("pdm_score_raw", predictions.get("pdm_score"))
    if expected is not None and (
        not torch.is_tensor(expected)
        or expected.shape != captured.shape[:2]
        or (not torch.isfinite(expected).all())
    ):
        raise ValueError("cached Stage3 scores must be finite [B,N]")
    initial_calls = 0
    if expected is None or (cfg.preflight_check and optimizer._batch_count == 0):
        output = optimizer._invoke_with_proposals(context, captured.detach())
        actual = _checked_scores(optimizer, context, output, captured.shape[:2])
        initial_calls = 1
        if expected is not None and (
            not torch.allclose(
                expected.detach().to(actual), actual, rtol=0.0001, atol=1e-05
            )
        ):
            raise RuntimeError(
                "goal-particle scorer preflight did not reproduce Stage3"
            )
        if expected is None:
            expected = actual
    original_scores = expected.detach().to(device=original.device, dtype=torch.float32)
    ego = features.get(cfg.ego_status_key)
    if (
        not torch.is_tensor(ego)
        or ego.ndim != 3
        or ego.shape[0] != batch
        or (ego.shape[1] < 3)
        or (ego.shape[2] < 7)
        or (not torch.isfinite(ego[:, -3:, :7]).all())
    ):
        raise ValueError(
            "goal_particle_rediffusion requires finite ego_status [B,H>=3,F>=7]"
        )
    return (
        context,
        original,
        reference,
        original_scores,
        slots,
        rows,
        ego[:, -3:, :7],
        initial_calls,
    )


def _run_v13_incumbent(
    optimizer,
    context,
    original,
    reference,
    original_scores,
    slots,
    ego_history,
    proxy,
    sample_ids,
):
    """Run the released v0.13 core and its unchanged final acceptance rule."""
    from navsim.tools.trajectory_mining.path_endpoint_surface import optimize_path_endpoint_surface

    cfg = optimizer.config
    batch = original.shape[0]
    cpu = torch.device("cpu")
    generators = make_sample_generators(cpu, optimizer._sample_seeds(sample_ids, batch))

    def score_cpu(completed, valid):
        if not bool(valid.any()):
            return torch.full(valid.shape, -torch.inf, device=cpu)
        return proxy(
            completed.to(device=original.device), valid.to(device=original.device)
        ).to(cpu)

    result = optimize_path_endpoint_surface(
        reference.float().to(cpu),
        original_scores.to(cpu),
        original.detach().float().to(cpu),
        score_cpu,
        config=cfg,
        generators=generators,
        ego_history=ego_history.to(device=cpu, dtype=torch.float32),
        output_dtype=original.dtype,
    )
    candidates = result.candidates.to(original)
    available = result.available.to(device=original.device)
    parents = result.parents.to(device=original.device)
    verified = _verify_shortlist(
        optimizer,
        context,
        reference,
        original,
        slots,
        candidates,
        available,
        original_scores,
    )
    rows = torch.arange(original.shape[0], device=original.device)
    chosen = verified["chosen"]
    final = original.clone()
    if candidates.shape[1]:
        final = torch.where(
            verified["switched"][:, None, None],
            candidates[rows, chosen.clamp_min(0)],
            original,
        )
    return (final, result, candidates, available, parents, verified)


def _terminal_speed(trajectory: torch.Tensor, dt: float) -> torch.Tensor:
    return (
        torch.linalg.vector_norm(
            trajectory[..., -1, :2].float() - trajectory[..., -2, :2].float(), dim=-1
        )
        / dt
    )


def _latent_flow_arms(optimizer, original_pool, particle, ego_history):
    """Decode raw/corrected latent arms; direct candidates stay independent."""
    cfg = optimizer.config
    batch, proposal_count = original_pool.shape[:2]
    if not cfg.goal_particle_rediffusion or particle.candidates.shape[1] == 0:
        empty = original_pool[:, :0]
        return (
            empty,
            torch.zeros(batch, 0, dtype=torch.bool, device=original_pool.device),
            {"enabled": False},
        )
    particle_available = particle.available.to(device=original_pool.device)
    if not bool(particle_available.any()):
        empty = original_pool[:, :0]
        return (
            empty,
            torch.zeros(batch, 0, dtype=torch.bool, device=original_pool.device),
            {"enabled": True, "skipped": "no_available_particle"},
        )
    from navsim.tools.trajectory_mining.frozen_rediffusion import FrozenRediffusionAdapter

    count = min(cfg.goal_particle_rediffusion_topk, particle.candidates.shape[1])
    merit = particle.effective_scores.to(device=original_pool.device).masked_fill(
        ~particle_available, -torch.inf
    )
    top = merit.topk(count, dim=1).indices
    rows = torch.arange(batch, device=original_pool.device)[:, None]
    direct = particle.candidates.to(original_pool)[rows, top]
    selected_available = particle_available[rows, top]
    parents_source = particle.parents.to(device=original_pool.device)
    selected_parents_raw = parents_source[rows, top]
    invalid_parent = (selected_parents_raw < 0) | (
        selected_parents_raw >= proposal_count
    )
    if bool((selected_available & invalid_parent).any()):
        raise RuntimeError("active goal-particle candidate has an invalid parent slot")
    selected_parents = torch.where(
        selected_available, selected_parents_raw, torch.zeros_like(selected_parents_raw)
    )
    selected_query_indices = particle.details.get("selected_query_indices")
    achieved_weights = particle.details.get("query_achieved_weights")
    if (
        not torch.is_tensor(selected_query_indices)
        or selected_query_indices.shape != particle.available.shape
        or (not torch.is_tensor(achieved_weights))
        or (achieved_weights.ndim != 4)
        or (achieved_weights.shape[0] != batch)
        or (achieved_weights.shape[-1] != proposal_count)
    ):
        raise RuntimeError(
            "goal-particle result is missing shortlist achieved-support provenance"
        )
    flat_weights = achieved_weights.to(device=original_pool.device).flatten(1, 2)
    query_indices_raw = selected_query_indices.to(device=original_pool.device)[
        rows, top
    ]
    invalid_query = (query_indices_raw < 0) | (
        query_indices_raw >= flat_weights.shape[1]
    )
    if bool((selected_available & invalid_query).any()):
        raise RuntimeError(
            "active goal-particle candidate has invalid support provenance"
        )
    query_indices = torch.where(
        selected_available, query_indices_raw, torch.zeros_like(query_indices_raw)
    )
    support_weights = flat_weights[rows, query_indices]
    fallback_support = torch.zeros(
        batch, count, proposal_count, device=original_pool.device, dtype=torch.float32
    )
    fallback_support.scatter_(2, selected_parents[..., None], 1.0)
    support_weights = torch.where(
        selected_available[..., None], support_weights.float(), fallback_support
    )
    adapter = FrozenRediffusionAdapter.from_captured(
        optimizer._captured_generator_context()
    )
    preflight = None
    preflight_key = (
        id(adapter.module),
        str(adapter.captured_ar_proposals.device),
        str(adapter.captured_ar_proposals.dtype),
        tuple(adapter.captured_ar_proposals.shape[1:]),
    )
    if (
        cfg.goal_particle_rediffusion_preflight
        and getattr(optimizer, "_goal_particle_latent_preflight_key", None)
        != preflight_key
    ):
        preflight = adapter.replay_preflight(original_pool)
        optimizer._goal_particle_latent_preflight_key = preflight_key
    transported = adapter.latent_flow_transport(direct, original_pool, support_weights)
    raw = transported.raw.to(original_pool)
    corrected = transported.corrected.to(original_pool)
    linear_final = transported.linear_final.to(original_pool)

    def audit_arm(arm):
        finite = torch.isfinite(arm).flatten(2).all(-1)
        endpoint_error = torch.linalg.vector_norm(
            arm[..., -1, :2].float() - direct[..., -1, :2].float(), dim=-1
        )
        support_ade = torch.linalg.vector_norm(
            arm[..., :2].float() - linear_final[..., :2].float(), dim=-1
        ).mean(-1)
        support_endpoint = torch.linalg.vector_norm(
            arm[..., -1, :2].float() - linear_final[..., -1, :2].float(), dim=-1
        )
        support_heading = (
            torch.atan2(
                torch.sin(arm[..., 2].float() - linear_final[..., 2].float()),
                torch.cos(arm[..., 2].float() - linear_final[..., 2].float()),
            )
            .abs()
            .amax(-1)
        )
        support_speed = (
            _terminal_speed(arm, cfg.trajectory_dt)
            - _terminal_speed(linear_final, cfg.trajectory_dt)
        ).abs()
        support_ok = (
            (support_ade <= cfg.goal_particle_shadow_max_ade)
            & (support_endpoint <= cfg.goal_particle_shadow_max_endpoint)
            & (support_heading <= cfg.goal_particle_shadow_max_heading)
            & (support_speed <= cfg.goal_particle_shadow_max_speed)
        )
        dynamics_ok = torch.zeros_like(selected_available)
        dynamics_reason = torch.zeros_like(selected_parents)
        dynamics_metrics = torch.zeros(batch, count, 12, device=original_pool.device)
        dynamics_cost = torch.zeros(batch, count, device=original_pool.device)
        for batch_row in range(batch):
            requested = direct[batch_row, :, -1, :2].float()
            achieved = arm[batch_row, :, -1, :2].float()
            history = _history_summary(ego_history[batch_row].float(), cfg.history_dt)
            ok, reason, metrics, cost = _path_metrics(
                arm[batch_row],
                linear_final[batch_row],
                requested,
                achieved,
                history,
                cfg,
            )
            dynamics_ok[batch_row] = ok
            dynamics_reason[batch_row] = reason
            dynamics_metrics[batch_row] = metrics
            dynamics_cost[batch_row] = cost
        available = (
            selected_available
            & finite
            & (endpoint_error <= cfg.goal_particle_rediffusion_endpoint_tolerance)
            & support_ok
            & dynamics_ok
        )
        return (
            available,
            {
                "finite": finite,
                "endpoint_error": endpoint_error,
                "support_ade": support_ade,
                "support_endpoint": support_endpoint,
                "support_heading": support_heading,
                "support_speed": support_speed,
                "support_ok": support_ok,
                "dynamics_ok": dynamics_ok,
                "dynamics_reason": dynamics_reason,
                "dynamics_metrics": dynamics_metrics,
                "dynamics_cost": dynamics_cost,
                "actual_endpoints": arm[..., -1, :],
                "available": available,
            },
        )

    raw_available, raw_audit = audit_arm(raw)
    corrected_available, corrected_audit = audit_arm(corrected)
    arms = torch.cat((raw, corrected), dim=1)
    available = torch.cat((raw_available, corrected_available), dim=1)
    return (
        arms,
        available,
        {
            "enabled": True,
            "selected_particle_indices": top,
            "selected_available": selected_available,
            "selected_query_indices": query_indices_raw,
            "parents": selected_parents,
            "direct": direct,
            "support_weights": transported.support_weights,
            "effective_support": transported.effective_support,
            "variance_scale": transported.variance_scale,
            "noise_scale": transported.noise_scale,
            "linear_ar": transported.linear_ar,
            "linear_final": transported.linear_final,
            "mixed_noise": transported.mixed_noise,
            "raw": raw,
            "corrected": corrected,
            "raw_audit": raw_audit,
            "corrected_audit": corrected_audit,
            "available": available,
            "preflight": vars(preflight) if preflight is not None else None,
        },
    )
