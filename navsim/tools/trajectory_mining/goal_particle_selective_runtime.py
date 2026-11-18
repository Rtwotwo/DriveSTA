"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
import torch
from navsim.tools.trajectory_mining.cem import make_sample_generators
from navsim.tools.trajectory_mining.endpoint_runtime import PackedEndpointScorer, _checked_scores, _scorer_dtype, _encode, _row
from navsim.tools.trajectory_mining.goal_particle_runtime import _latent_flow_arms, _prepare_inputs, _run_v13_incumbent
from navsim.tools.trajectory_mining.replacement import _component_probabilities, _replacement_call
import gzip
import json
import os
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _geometric_progress(
    incumbent: torch.Tensor, candidates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return terminal forward and path-length changes relative to incumbent.

    The terminal direction is taken from the incumbent's final heading.  This
    is an intentionally weak, scene-free progress diagnostic, not a route or
    safety oracle.  Path length includes the current ego origin.
    """
    batch, count = candidates.shape[:2]
    heading = incumbent[:, -1, 2].float()
    axis = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    endpoint_delta = (
        candidates[..., -1, :2].float() - incumbent[:, None, -1, :2].float()
    )
    forward = (endpoint_delta * axis[:, None]).sum(-1)
    origin = torch.zeros(batch, 1, 2, device=incumbent.device, dtype=torch.float32)
    incumbent_xy = torch.cat((origin, incumbent[..., :2].float()), dim=1)
    candidate_origin = origin[:, None].expand(-1, count, -1, -1)
    candidate_xy = torch.cat((candidate_origin, candidates[..., :2].float()), dim=2)
    incumbent_length = torch.linalg.vector_norm(
        incumbent_xy[:, 1:] - incumbent_xy[:, :-1], dim=-1
    ).sum(-1)
    candidate_length = torch.linalg.vector_norm(
        candidate_xy[:, :, 1:] - candidate_xy[:, :, :-1], dim=-1
    ).sum(-1)
    return (forward, candidate_length - incumbent_length[:, None])


def _selective_compare(
    optimizer,
    context,
    reference,
    incumbent,
    anchor_slots,
    candidates,
    available,
    source_enabled,
    source_names,
    corrected_mask,
):
    """Reproduce five-arm ranking, then veto unreliable winning sources."""
    cfg = optimizer.config
    batch, proposal_count = reference.shape[:2]
    count = candidates.shape[1]
    contexts = min(cfg.goal_particle_robust_contexts, proposal_count - 1)
    if contexts < 2:
        raise ValueError("goal_particle_selective needs at least two proposal contexts")
    if source_enabled.dtype != torch.bool or source_enabled.shape != (count,):
        raise ValueError("source_enabled must be boolean [candidate_count]")
    rows = torch.arange(batch, device=reference.device)
    incumbent_reference = reference.clone()
    incumbent_reference[rows, anchor_slots] = incumbent.to(reference)
    slot_ids = torch.arange(proposal_count, device=reference.device)
    non_anchor = slot_ids.expand(batch, -1)
    non_anchor = non_anchor[non_anchor != anchor_slots[:, None]].reshape(
        batch, proposal_count - 1
    )
    positions = (
        torch.linspace(0, proposal_count - 2, contexts, device=reference.device)
        .round()
        .long()
    )
    candidate_slots = non_anchor[:, positions]
    gains = torch.full((batch, count, contexts), -torch.inf, device=reference.device)
    is_argmax = torch.zeros_like(gains, dtype=torch.bool)
    safety = torch.zeros_like(gains, dtype=torch.bool)
    safety_pairs = {
        key: torch.full((batch, count, contexts, 2), torch.nan, device=reference.device)
        for key in cfg.goal_particle_safety_keys
    }
    progress_pairs = torch.full(
        (batch, count, contexts, 2), torch.nan, device=reference.device
    )
    progress_seen = False
    calls = 0
    evaluations = torch.zeros(batch, dtype=torch.long, device=reference.device)
    flat = available.flatten().nonzero(as_tuple=False).flatten()
    for view in range(contexts):
        for start in range(0, flat.numel(), cfg.replacement_chunk_size):
            ids = flat[start : start + cfg.replacement_chunk_size]
            sample_rows = torch.div(ids, count, rounding_mode="floor")
            candidate_ids = ids.remainder(count)
            pool = incumbent_reference.index_select(0, sample_rows).clone()
            local_rows = torch.arange(ids.numel(), device=reference.device)
            local_slots = candidate_slots[sample_rows, view]
            pool[local_rows, local_slots] = candidates[sample_rows, candidate_ids].to(
                pool
            )
            output = _replacement_call(
                context.module,
                context.args,
                context.kwargs,
                pool.to(dtype=_scorer_dtype(context, pool)),
                sample_rows,
                batch,
            )
            scores = _checked_scores(optimizer, context, output, pool.shape[:2])
            components = _component_probabilities(output, tuple(pool.shape[:2]))
            missing = set(cfg.goal_particle_safety_keys) - components.keys()
            if missing:
                raise RuntimeError(
                    f"goal-particle selective scorer is missing safety logits: {sorted(missing)}"
                )
            candidate_score = scores[local_rows, local_slots]
            incumbent_score = scores[local_rows, anchor_slots[sample_rows]]
            gains[sample_rows, candidate_ids, view] = candidate_score - incumbent_score
            is_argmax[sample_rows, candidate_ids, view] = scores.argmax(1).eq(
                local_slots
            )
            view_safe = torch.ones(
                ids.numel(), dtype=torch.bool, device=reference.device
            )
            for key in cfg.goal_particle_safety_keys:
                candidate_component = components[key][local_rows, local_slots]
                incumbent_component = components[key][
                    local_rows, anchor_slots[sample_rows]
                ]
                safety_pairs[key][
                    sample_rows, candidate_ids, view, 0
                ] = candidate_component
                safety_pairs[key][
                    sample_rows, candidate_ids, view, 1
                ] = incumbent_component
                view_safe &= (
                    candidate_component + cfg.goal_particle_safety_tolerance
                    >= incumbent_component
                )
            safety[sample_rows, candidate_ids, view] = view_safe
            progress_key = cfg.goal_particle_selective_progress_key
            if progress_key in components:
                progress_seen = True
                progress_pairs[sample_rows, candidate_ids, view, 0] = components[
                    progress_key
                ][local_rows, local_slots]
                progress_pairs[sample_rows, candidate_ids, view, 1] = components[
                    progress_key
                ][local_rows, anchor_slots[sample_rows]]
            evaluations.scatter_add_(0, sample_rows, torch.ones_like(sample_rows))
            calls += 1
    min_context_gain = gains.amin(-1)
    finite_gains = torch.where(torch.isfinite(gains), gains, torch.zeros_like(gains))
    median_gain = finite_gains.median(-1).values
    mad_gain = (finite_gains - median_gain[..., None]).abs().median(-1).values
    win = (gains > 0) & is_argmax
    win_fraction = win.float().mean(-1)
    safety_fraction = safety.float().mean(-1)
    safety_all = safety.all(-1)
    changed = (candidates != incumbent[:, None]).flatten(2).any(-1)
    rescue_by_key = {}
    for key in cfg.goal_particle_selective_rescue_keys:
        delta = safety_pairs[key][..., 0] - safety_pairs[key][..., 1]
        rescue_by_key[key] = (
            (delta > cfg.goal_particle_selective_rescue_margin).float().mean(-1)
        )
    rescue_fraction = torch.stack(tuple(rescue_by_key.values()), dim=-1).amax(-1)
    rescue_evidence = rescue_fraction >= cfg.goal_particle_robust_min_fraction
    progress_delta = progress_pairs[..., 0] - progress_pairs[..., 1]
    progress_finite = torch.isfinite(progress_delta).all(-1)
    progress_fraction = (
        (progress_delta >= -cfg.goal_particle_selective_progress_tolerance)
        .float()
        .mean(-1)
    )
    forward_delta, path_length_delta = _geometric_progress(incumbent, candidates)
    geometry_progress_ok = (
        forward_delta >= -cfg.goal_particle_selective_forward_tolerance
    ) & (path_length_delta >= -cfg.goal_particle_selective_path_length_tolerance)
    normal_progress_ok = (
        progress_finite
        & (progress_fraction >= cfg.goal_particle_robust_min_fraction)
        & geometry_progress_ok
    )
    if (
        cfg.goal_particle_selective_dual_gate
        and bool(available.any())
        and (not progress_seen)
    ):
        raise RuntimeError(
            f"goal_particle_selective dual gate requires scorer component {cfg.goal_particle_selective_progress_key!r}"
        )
    dual_gate_ok = rescue_evidence | normal_progress_ok
    if not cfg.goal_particle_selective_dual_gate:
        dual_gate_ok = torch.ones_like(dual_gate_ok)
    corrected_source_ok = ~corrected_mask | rescue_evidence
    if not cfg.goal_particle_selective_corrected_rescue_only:
        corrected_source_ok = torch.ones_like(corrected_source_ok)
    common = (
        available
        & changed
        & (min_context_gain > cfg.goal_particle_score_margin)
        & (win_fraction >= cfg.goal_particle_robust_min_fraction)
        & safety_all
    )
    pre_veto_eligible = common & dual_gate_ok & corrected_source_ok
    merit = min_context_gain.masked_fill(~pre_veto_eligible, -torch.inf)
    pre_veto_chosen = (
        merit.argmax(1)
        if count
        else torch.zeros(batch, device=reference.device, dtype=torch.long)
    )
    pre_veto_switched = pre_veto_eligible.any(1)
    pre_veto_chosen = torch.where(pre_veto_switched, pre_veto_chosen, -1)
    winner_source_allowed = (
        source_enabled[pre_veto_chosen.clamp_min(0)]
        if count
        else torch.zeros(batch, dtype=torch.bool, device=reference.device)
    )
    source_vetoed = pre_veto_switched & ~winner_source_allowed
    switched = pre_veto_switched & winner_source_allowed
    chosen = torch.where(switched, pre_veto_chosen, -1)
    pre_veto_selected_gain = (
        torch.where(
            pre_veto_switched,
            min_context_gain[rows, pre_veto_chosen.clamp_min(0)],
            torch.zeros_like(min_context_gain[:, 0]),
        )
        if count
        else torch.zeros(batch, device=reference.device)
    )
    selected_gain = torch.where(
        switched, pre_veto_selected_gain, torch.zeros_like(pre_veto_selected_gain)
    )
    eligible = pre_veto_eligible & source_enabled[None]
    rejection = torch.zeros(batch, count, dtype=torch.long, device=reference.device)
    rejection = torch.where(source_enabled[None], rejection, 1)
    rejection = torch.where(source_enabled[None] & ~available, 2, rejection)
    rejection = torch.where(available & ~changed, 3, rejection)
    rejection = torch.where(
        available & changed & ~(min_context_gain > cfg.goal_particle_score_margin),
        4,
        rejection,
    )
    rejection = torch.where(
        available
        & changed
        & (min_context_gain > cfg.goal_particle_score_margin)
        & ~safety_all,
        5,
        rejection,
    )
    rejection = torch.where(
        available
        & changed
        & (min_context_gain > cfg.goal_particle_score_margin)
        & safety_all
        & (win_fraction < cfg.goal_particle_robust_min_fraction),
        6,
        rejection,
    )
    rejection = torch.where(common & ~dual_gate_ok, 7, rejection)
    rejection = torch.where(common & dual_gate_ok & ~corrected_source_ok, 8, rejection)
    rejection = torch.where(~source_enabled[None], 1, rejection)
    pre_veto_selected_source = [
        source_names[int(index)] if int(index) >= 0 else "v13_fallback"
        for index in pre_veto_chosen.detach().cpu().tolist()
    ]
    selected_source = [
        source_names[int(index)] if int(index) >= 0 else "v13_fallback"
        for index in chosen.detach().cpu().tolist()
    ]
    return {
        "aggregation": "min_context",
        "source_names": source_names,
        "source_enabled": source_enabled,
        "corrected_mask": corrected_mask,
        "gains": gains,
        "min_context_gain": min_context_gain,
        "median_gain": median_gain,
        "mad_gain_diagnostic_only": mad_gain,
        "is_argmax": is_argmax,
        "win_fraction": win_fraction,
        "safety": safety,
        "safety_fraction": safety_fraction,
        "safety_all": safety_all,
        "components": safety_pairs,
        "progress_pairs": progress_pairs,
        "progress_seen": progress_seen,
        "progress_fraction": progress_fraction,
        "forward_delta": forward_delta,
        "path_length_delta": path_length_delta,
        "geometry_progress_ok": geometry_progress_ok,
        "normal_progress_ok": normal_progress_ok,
        "rescue_fraction_by_key": rescue_by_key,
        "rescue_fraction": rescue_fraction,
        "rescue_evidence": rescue_evidence,
        "dual_gate_ok": dual_gate_ok,
        "corrected_source_ok": corrected_source_ok,
        "pre_veto_eligible": pre_veto_eligible,
        "pre_veto_chosen": pre_veto_chosen,
        "pre_veto_selected_source": pre_veto_selected_source,
        "pre_veto_switched": pre_veto_switched,
        "pre_veto_selected_gain": pre_veto_selected_gain,
        "winner_source_allowed": winner_source_allowed,
        "source_vetoed": source_vetoed,
        "eligible": eligible,
        "chosen": chosen,
        "selected_source": selected_source,
        "switched": switched,
        "selected_gain": selected_gain,
        "rejection_code": rejection,
        "scorer_calls": calls,
        "pool_evaluations": evaluations,
        "candidate_slots": candidate_slots,
    }


def _write_diagnostics(
    optimizer,
    sample_ids,
    reference,
    original_scores,
    original,
    final,
    anchor_slots,
    v13_result,
    v13_candidates,
    v13_available,
    v13_verified,
    particle,
    latent_flow_candidates,
    latent_flow_generated_available,
    latent_flow_details,
    new_candidates,
    new_available,
    selective,
    proxy,
    initial_calls,
):
    from navsim.tools.trajectory_mining import __version__

    if optimizer._diagnostic_path is None:
        directory = Path(optimizer.config.diagnostics_dir)
        directory.mkdir(parents=True, exist_ok=True)
        optimizer._diagnostic_path = (
            directory
            / f"goal_particle_selective_{os.getpid()}_{time.time_ns()}.jsonl.gz"
        )
    batch = reference.shape[0]
    lines = []
    for row in range(batch):
        token = str(sample_ids[row]) if sample_ids is not None else None
        selective_row = _row(
            {
                key: value
                for key, value in selective.items()
                if key
                not in {
                    "source_names",
                    "source_enabled",
                    "corrected_mask",
                    "selected_source",
                    "pre_veto_selected_source",
                }
            },
            row,
            batch,
        )
        selective_row["source_names"] = selective["source_names"]
        selective_row["source_enabled"] = selective["source_enabled"]
        selective_row["corrected_mask"] = selective["corrected_mask"]
        selective_row["pre_veto_selected_source"] = selective[
            "pre_veto_selected_source"
        ][row]
        selective_row["selected_source"] = selective["selected_source"][row]
        record = {
            "schema": "chainflow_tto_goal_particle_selective_v20_v1",
            "package_version": __version__,
            "strategy": "goal_particle_selective",
            "sample_id": token,
            "token": token,
            "batch_index": optimizer._batch_count,
            "training_used": False,
            "configuration": optimizer.config.to_dict(),
            "original_proposals": reference[row],
            "original_scores": original_scores[row],
            "original_trajectory": original[row],
            "original_anchor_slot": anchor_slots[row],
            "v13_candidates": v13_candidates[row],
            "v13_available": v13_available[row],
            "v13_search": _row(v13_result.details, row, batch),
            "v13_verification": _row(v13_verified, row, batch),
            "particle_candidates": particle.candidates[row],
            "particle_available": particle.available[row],
            "particle_parents": particle.parents[row],
            "particle_requested_states": particle.requested_states[row],
            "particle_achieved_states": particle.achieved_states[row],
            "particle_raw_scores": particle.raw_scores[row],
            "particle_effective_scores": particle.effective_scores[row],
            "particle_search": _row(particle.details, row, batch),
            "latent_flow_candidates": latent_flow_candidates[row],
            "latent_flow_generated_available": latent_flow_generated_available[row],
            "latent_flow": _row(latent_flow_details, row, batch),
            "new_candidates": new_candidates[row],
            "new_available": new_available[row],
            "selective_verification": selective_row,
            "final_trajectory": final[row],
            "initial_scorer_calls": initial_calls,
            "proxy_scorer_calls": proxy.scorer_calls,
            "proxy_scored_slots": proxy.scored_slots[row],
            "proxy_pool_evaluations": proxy.pool_evaluations[row],
            "selection_boundary": "the exact five-arm v0.19 winner replaces v0.13 only after all-context safety/min-context gates and the source veto",
            "calibration_boundary": "dual rescue/progress features are diagnostic-only by default; enable their gate only after recording-disjoint held-out calibration",
            "safety_boundary": "predicted scorer components plus dynamics/support checks; not a map, actor, metric-cache, or true-PDM certificate",
            "rejection_codes": {
                "0": "eligible",
                "1": "source_disabled",
                "2": "unavailable",
                "3": "unchanged",
                "4": "min_context_margin",
                "5": "predicted_safety_not_all_contexts",
                "6": "argmax_win_fraction",
                "7": "optional_dual_gate",
                "8": "corrected_requires_rescue",
            },
        }
        lines.append(
            json.dumps(_encode(record), allow_nan=False, separators=(",", ":")) + "\n"
        )
    with gzip.open(
        optimizer._diagnostic_path, "at", encoding="utf-8", compresslevel=3
    ) as stream:
        stream.writelines(lines)


@torch.no_grad()
def refine_goal_particle_selective_predictions(
    optimizer, predictions, features, sample_ids
):
    """Run exact v0.13 + unchanged v0.19 generation + v0.20 selection."""
    from navsim.tools.trajectory_mining.goal_particle_surface import optimize_goal_particle_surface

    cfg = optimizer.config
    (
        context,
        original,
        reference,
        original_scores,
        slots,
        rows,
        ego_history,
        initial_calls,
    ) = _prepare_inputs(optimizer, predictions, features, sample_ids)
    proxy = PackedEndpointScorer(optimizer, context, reference, slots, original_scores)
    incumbent, v13_result, v13_candidates, v13_available, _v13_parents, v13_verified = (
        _run_v13_incumbent(
            optimizer,
            context,
            original,
            reference,
            original_scores,
            slots,
            ego_history,
            proxy,
            sample_ids,
        )
    )
    cpu = torch.device("cpu")
    particle_seeds = [
        (value ^ 1848019772081597199) & (1 << 63) - 1
        for value in optimizer._sample_seeds(sample_ids, original.shape[0])
    ]
    generators = make_sample_generators(cpu, particle_seeds)

    def score_cpu(completed, valid):
        if not bool(valid.any()):
            return torch.full(valid.shape, -torch.inf, device=cpu)
        return proxy(
            completed.to(device=original.device), valid.to(device=original.device)
        ).to(cpu)

    particle = optimize_goal_particle_surface(
        reference.float().to(cpu),
        original_scores.to(cpu),
        original.detach().float().to(cpu),
        score_cpu,
        config=cfg,
        generators=generators,
        ego_history=ego_history.to(device=cpu, dtype=torch.float32),
        output_dtype=original.dtype,
    )
    direct = particle.candidates.to(original)
    direct_available = particle.available.to(device=original.device).clone()
    direct_topk = min(cfg.goal_particle_selective_direct_topk, direct.shape[1])
    if cfg.goal_particle_selective_include_corrected:
        latent_flow_candidates, latent_flow_available, latent_flow_details = (
            _latent_flow_arms(optimizer, reference, particle, ego_history)
        )
    else:
        latent_flow_candidates = reference[:, :0]
        latent_flow_available = torch.zeros(
            original.shape[0], 0, dtype=torch.bool, device=original.device
        )
        latent_flow_details = {"enabled": False, "skipped": "selective_disabled"}
    latent_flow_generated_available = latent_flow_available.clone()
    latent_flow_available = latent_flow_generated_available.clone()
    latent_count = latent_flow_candidates.shape[1]
    raw_count = latent_count // 2
    new_candidates = torch.cat((direct, latent_flow_candidates), dim=1)
    new_available = torch.cat((direct_available, latent_flow_available), dim=1)
    source_names = [f"direct_{index}" for index in range(direct.shape[1])]
    source_names += [f"latent_raw_{index}" for index in range(raw_count)]
    source_names += [
        f"latent_corrected_{index}" for index in range(latent_count - raw_count)
    ]
    source_enabled = torch.zeros(
        new_candidates.shape[1], dtype=torch.bool, device=original.device
    )
    source_enabled[:direct_topk] = True
    if latent_count > raw_count:
        source_enabled[direct.shape[1] + raw_count :] = True
    corrected_mask = torch.zeros_like(source_enabled)
    if latent_count > raw_count:
        corrected_mask[direct.shape[1] + raw_count :] = True
    selective = _selective_compare(
        optimizer,
        context,
        reference,
        incumbent,
        slots,
        new_candidates,
        new_available,
        source_enabled,
        source_names,
        corrected_mask,
    )
    final = incumbent.clone()
    if new_candidates.shape[1]:
        final = torch.where(
            selective["switched"][:, None, None],
            new_candidates[rows, selective["chosen"].clamp_min(0)],
            incumbent,
        )
    updates = {
        cfg.trajectory_key: final,
        "tto_used": (final != original).flatten(1).any(-1),
        "tto_goal_particle_used": selective["switched"],
        "tto_goal_particle_chosen": selective["chosen"],
        "tto_goal_particle_min_context_gain": selective["selected_gain"],
        "tto_goal_particle_robust_gain": selective["selected_gain"],
        "tto_goal_particle_eligible_count": selective["eligible"].sum(1),
        "tto_goal_particle_latent_flow_available_count": latent_flow_available.sum(1),
        "tto_goal_particle_latent_flow_generated_available_count": latent_flow_generated_available.sum(
            1
        ),
        "tto_goal_particle_v13_used": v13_verified["switched"],
        "tto_goal_particle_v13_gain": v13_verified["selected_gain"],
        "tto_delta_l2": (final.float()[..., :2] - original.float()[..., :2])
        .norm(dim=-1)
        .mean(-1),
        "tto_goal_particle_initial_calls": initial_calls,
        "tto_goal_particle_proxy_calls": proxy.scorer_calls,
        "tto_goal_particle_v13_verification_calls": v13_verified["scorer_calls"],
        "tto_goal_particle_robust_verification_calls": selective["scorer_calls"],
        "tto_goal_particle_source_vetoed": selective["source_vetoed"],
        "tto_goal_particle_pre_veto_chosen": selective["pre_veto_chosen"],
        "tto_goal_particle_pre_veto_gain": selective["pre_veto_selected_gain"],
    }
    if cfg.diagnostics_dir:
        _write_diagnostics(
            optimizer,
            sample_ids,
            reference,
            original_scores,
            original,
            final,
            slots,
            v13_result,
            v13_candidates,
            v13_available,
            v13_verified,
            particle,
            latent_flow_candidates,
            latent_flow_generated_available,
            latent_flow_details,
            new_candidates,
            new_available,
            selective,
            proxy,
            initial_calls,
        )
    optimizer._batch_count += 1
    if (
        cfg.log_every_n_batches
        and optimizer._batch_count % cfg.log_every_n_batches == 0
    ):
        logger.info(
            "goal_particle_selective batch=%s v13=%s/%s new=%s/%s proxy/final_calls=%s/%s",
            optimizer._batch_count,
            int(v13_verified["switched"].sum()),
            original.shape[0],
            int(selective["switched"].sum()),
            original.shape[0],
            proxy.scorer_calls,
            selective["scorer_calls"],
        )
    predictions.update(updates)
    return predictions
