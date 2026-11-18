"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
import torch
from navsim.tools.trajectory_mining.replacement import _component_probabilities, _replacement_call
import math

SAFETY_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)


def _checked_scores(optimizer, context, output, shape):
    scores = optimizer._stable_drivesta_score(context.module, output)
    if not torch.is_tensor(scores) or tuple(scores.shape) != tuple(shape):
        raise ValueError(
            f"endpoint scorer must return scores with shape {tuple(shape)}"
        )
    scores = scores.float()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite endpoint scorer scores")
    return scores


def _scorer_dtype(context, reference):
    parameter = next(context.module.parameters(), None)
    return parameter.dtype if parameter is not None else reference.dtype


class PackedEndpointScorer:
    """Cheap proxy preserving fixed reference and the original anchor slot.

    Candidate scores use ``original_anchor + candidate_now - anchor_now``. This
    removes a common context offset, NOT candidate-specific attention effects.
    Inactive slots retain their exact original proposal. No old PCMR is called.
    """

    def __init__(self, optimizer, context, reference, anchor_slots, original_scores):
        self.optimizer, self.context = (optimizer, context)
        self.reference, self.anchor_slots = (reference, anchor_slots)
        self.rows = torch.arange(reference.shape[0], device=reference.device)
        self.anchor_score = original_scores[self.rows, anchor_slots]
        count = min(
            8,
            optimizer.config.endpoint_modes
            * optimizer.config.endpoint_samples_per_mode,
            reference.shape[1] - 1,
        )
        slot_ids = torch.arange(reference.shape[1], device=reference.device)
        self.slots = slot_ids.expand(reference.shape[0], -1)
        self.slots = self.slots[self.slots != anchor_slots[:, None]].reshape(
            reference.shape[0], -1
        )[:, :count]
        self.scorer_calls = 0
        self.scored_slots = torch.zeros(
            reference.shape[0], dtype=torch.long, device=reference.device
        )
        self.pool_evaluations = torch.zeros_like(self.scored_slots)
        self.traces = []

    def __call__(self, candidates, valid):
        batch, _, poses, _ = self.reference.shape
        if (
            candidates.ndim != 4
            or candidates.shape[0] != batch
            or candidates.shape[2:] != (poses, 3)
        ):
            raise ValueError("endpoint proxy candidates must have shape [B,Q,T,3]")
        if valid.dtype != torch.bool or valid.shape != candidates.shape[:2]:
            raise ValueError("endpoint proxy mask must be boolean [B,Q]")
        if not torch.isfinite(candidates[valid]).all():
            raise FloatingPointError("non-finite active endpoint proxy candidate")
        count = candidates.shape[1]
        scores = torch.full((batch, count), -torch.inf, device=self.reference.device)
        width = self.slots.shape[1]
        if width == 0:
            return scores
        executable = candidates.to(self.reference)
        for start in range(0, count, width):
            stop = min(start + width, count)
            mask = valid[:, start:stop]
            if not bool(mask.any()):
                continue
            pool = self.reference.clone()
            slots = self.slots[:, : stop - start]
            row_idx = self.rows[:, None].expand_as(slots)
            old = pool[row_idx, slots]
            pool[row_idx, slots] = torch.where(
                mask[..., None, None], executable[:, start:stop], old
            )
            output = self.optimizer._invoke_with_proposals(
                self.context, pool.to(dtype=_scorer_dtype(self.context, pool))
            )
            result = _checked_scores(
                self.optimizer, self.context, output, pool.shape[:2]
            )
            relative = (
                result[row_idx, slots] - result[self.rows, self.anchor_slots, None]
            )
            calibrated = relative + self.anchor_score[:, None]
            scores[:, start:stop] = torch.where(mask, calibrated, -torch.inf)
            self.scorer_calls += 1
            self.scored_slots += mask.sum(1)
            self.pool_evaluations += 1
        if self.optimizer.config.diagnostics_dir:
            self.traces.append(
                {
                    "trajectories": executable.detach().cpu(),
                    "valid": valid.detach().cpu(),
                    "proxy_scores": scores.detach().cpu(),
                }
            )
        return scores


def _verify_shortlist(
    optimizer,
    context,
    reference,
    original,
    anchor_slots,
    candidates,
    available,
    original_scores,
):
    """One original pool per candidate; original anchor remains in its slot.

    The candidate always occupies the same non-anchor slot within each scene. The
    candidate and anchor scores/components therefore share the actual final context.
    This checks frozen-scorer preference, not true safety or true PDM improvement.
    """
    batch, proposal_count = reference.shape[:2]
    count = candidates.shape[1]
    rows = torch.arange(batch, device=reference.device)
    candidate_slots = torch.where(anchor_slots == 0, 1, 0)
    scores = torch.full((batch, count), -torch.inf, device=reference.device)
    anchor_scores = torch.full_like(scores, torch.nan)
    is_argmax = torch.zeros((batch, count), dtype=torch.bool, device=reference.device)
    component_pairs = {
        key: torch.full((batch, count, 2), torch.nan, device=reference.device)
        for key in SAFETY_KEYS
    }
    flat_indices = available.flatten().nonzero(as_tuple=False).flatten()
    calls = 0
    evaluations = torch.zeros(batch, dtype=torch.long, device=reference.device)
    if proposal_count < 2 and bool(available.any()):
        raise ValueError("endpoint verification needs an anchor and a candidate slot")
    for start in range(
        0, flat_indices.numel(), optimizer.config.replacement_chunk_size
    ):
        flat = flat_indices[start : start + optimizer.config.replacement_chunk_size]
        sample_rows = torch.div(flat, count, rounding_mode="floor")
        candidate_ids = flat.remainder(count)
        pool = reference.index_select(0, sample_rows).clone()
        local_rows = torch.arange(flat.numel(), device=reference.device)
        local_slots = candidate_slots[sample_rows]
        pool[local_rows, local_slots] = candidates[sample_rows, candidate_ids].to(pool)
        output = _replacement_call(
            context.module,
            context.args,
            context.kwargs,
            pool.to(dtype=_scorer_dtype(context, pool)),
            sample_rows,
            batch,
        )
        result = _checked_scores(optimizer, context, output, pool.shape[:2])
        components = _component_probabilities(output, tuple(pool.shape[:2]))
        missing = set(SAFETY_KEYS) - components.keys()
        if missing:
            raise RuntimeError(
                f"endpoint final scorer is missing safety logits: {sorted(missing)}"
            )
        scores[sample_rows, candidate_ids] = result[local_rows, local_slots]
        anchor_scores[sample_rows, candidate_ids] = result[
            local_rows, anchor_slots[sample_rows]
        ]
        is_argmax[sample_rows, candidate_ids] = result.argmax(1).eq(local_slots)
        for key in SAFETY_KEYS:
            component_pairs[key][sample_rows, candidate_ids, 0] = components[key][
                local_rows, local_slots
            ]
            component_pairs[key][sample_rows, candidate_ids, 1] = components[key][
                local_rows, anchor_slots[sample_rows]
            ]
        evaluations.scatter_add_(0, sample_rows, torch.ones_like(sample_rows))
        calls += 1
    gains = scores - anchor_scores
    margin_ok = available & (gains > optimizer.config.endpoint_score_margin)
    safety_ok = available.clone()
    for pair in component_pairs.values():
        safety_ok &= (
            pair[..., 0] + optimizer.config.endpoint_safety_tolerance >= pair[..., 1]
        )
    changed = (candidates != original[:, None]).flatten(2).any(-1)
    eligible = available & margin_ok & safety_ok & changed
    if optimizer.config.endpoint_require_argmax:
        eligible &= is_argmax
    merit = torch.where(eligible, gains, -torch.inf)
    chosen = (
        merit.argmax(1)
        if count
        else torch.zeros(batch, device=reference.device, dtype=torch.long)
    )
    switched = eligible.any(1)
    chosen = torch.where(switched, chosen, -1)
    selected_gain = (
        torch.where(switched, gains[rows, chosen.clamp_min(0)], 0.0)
        if count
        else torch.zeros(batch, device=reference.device)
    )
    rejection = torch.full((batch, count), 0, dtype=torch.long, device=reference.device)
    rejection = torch.where(available, rejection, 1)
    rejection = torch.where(available & ~changed, 4, rejection)
    rejection = torch.where(available & changed & ~margin_ok, 2, rejection)
    rejection = torch.where(available & changed & margin_ok & ~safety_ok, 3, rejection)
    if optimizer.config.endpoint_require_argmax:
        rejection = torch.where(
            available & changed & margin_ok & safety_ok & ~is_argmax, 5, rejection
        )
    return {
        "scores": scores,
        "anchor_scores": anchor_scores,
        "gains": gains,
        "components": component_pairs,
        "eligible": eligible,
        "chosen": chosen,
        "switched": switched,
        "selected_gain": selected_gain,
        "rejection_code": rejection,
        "scorer_calls": calls,
        "is_argmax": is_argmax,
        "pool_evaluations": evaluations,
        "calibrated_scores": original_scores[rows, anchor_slots, None] + gains,
    }


def _encode(value):
    if torch.is_tensor(value):
        return _encode(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value


def _row(value, row, batch):
    if torch.is_tensor(value) and value.ndim and (value.shape[0] == batch):
        return value[row]
    if isinstance(value, dict):
        return {key: _row(item, row, batch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_row(item, row, batch) for item in value]
    return value
