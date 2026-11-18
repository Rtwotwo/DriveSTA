"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
import math
from typing import Any, Mapping, Optional, Sequence
import torch
import torch.nn as nn
from dataclasses import dataclass


class RediffusionCompatibilityError(RuntimeError):
    """Raised when safe frozen decoder replay/transport cannot be guaranteed."""


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-pi, pi]`` without changing dtype or device."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _validate_trajectory_bank(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise RediffusionCompatibilityError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise RediffusionCompatibilityError(
            f"{name} must have shape [batch, modes, poses, state], got {tuple(value.shape)}"
        )
    if any((size <= 0 for size in value.shape[:3])):
        raise RediffusionCompatibilityError(
            f"{name} must have non-empty batch/mode/pose dimensions"
        )
    if value.shape[-1] < 3:
        raise RediffusionCompatibilityError(
            f"{name} needs at least x/y/heading, got state size {value.shape[-1]}"
        )
    if not value.is_floating_point():
        raise RediffusionCompatibilityError(f"{name} must use a floating dtype")
    if not torch.isfinite(value).all():
        raise RediffusionCompatibilityError(f"{name} contains non-finite values")


def _gather_parents(bank: torch.Tensor, parent_indices: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(parent_indices):
        raise RediffusionCompatibilityError("parent_indices must be a torch.Tensor")
    if parent_indices.ndim != 2:
        raise RediffusionCompatibilityError(
            f"parent_indices must have shape [batch, candidates], got {tuple(parent_indices.shape)}"
        )
    if parent_indices.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise RediffusionCompatibilityError("parent_indices must use an integer dtype")
    if parent_indices.shape[0] != bank.shape[0]:
        raise RediffusionCompatibilityError(
            "parent_indices batch size does not match the proposal bank"
        )
    indices = parent_indices.to(device=bank.device, dtype=torch.long)
    if indices.numel() and (
        bool((indices < 0).any()) or bool((indices >= bank.shape[1]).any())
    ):
        raise RediffusionCompatibilityError(
            f"parent_indices must be in [0, {bank.shape[1] - 1}]"
        )
    gather_index = indices[..., None, None].expand(-1, -1, bank.shape[2], bank.shape[3])
    return torch.gather(bank, dim=1, index=gather_index)


def _normalize_support_weights(
    support_weights: torch.Tensor, bank: torch.Tensor
) -> torch.Tensor:
    """Validate and normalize ``[B,K,M]`` non-negative proposal support."""
    if not torch.is_tensor(support_weights):
        raise RediffusionCompatibilityError("support_weights must be a torch.Tensor")
    if support_weights.ndim != 3:
        raise RediffusionCompatibilityError(
            "support_weights must have shape [batch, candidates, original_modes]"
        )
    if (
        support_weights.shape[0] != bank.shape[0]
        or support_weights.shape[2] != bank.shape[1]
    ):
        raise RediffusionCompatibilityError(
            "support_weights batch/original-mode dimensions do not match the proposal bank"
        )
    if not support_weights.is_floating_point():
        raise RediffusionCompatibilityError("support_weights must use a floating dtype")
    weights = support_weights.to(device=bank.device, dtype=torch.float32)
    if not torch.isfinite(weights).all():
        raise RediffusionCompatibilityError("support_weights contain non-finite values")
    if bool((weights < -1e-07).any()):
        raise RediffusionCompatibilityError("support_weights must be non-negative")
    weights = weights.clamp_min(0.0)
    mass = weights.sum(-1, keepdim=True)
    if bool((mass <= 1e-08).any()):
        raise RediffusionCompatibilityError(
            "every candidate must have positive original-proposal support"
        )
    return weights / mass


def _barycentric_trajectory_mix(
    bank: torch.Tensor, weights: torch.Tensor, *, heading_index: int
) -> torch.Tensor:
    """Mix a proposal bank while treating heading as a circular coordinate."""
    _validate_trajectory_bank("barycentric proposal bank", bank)
    if not 0 <= heading_index < bank.shape[-1]:
        raise RediffusionCompatibilityError(
            f"heading_index={heading_index} is outside the state dimension"
        )
    calculation_dtype = torch.float64 if bank.dtype == torch.float64 else torch.float32
    values = bank.to(dtype=calculation_dtype)
    local_weights = weights.to(device=bank.device, dtype=calculation_dtype)
    mixed = torch.einsum("bkm,bmtd->bktd", local_weights, values)
    heading = values[..., heading_index]
    sin_heading = torch.einsum("bkm,bmt->bkt", local_weights, torch.sin(heading))
    cos_heading = torch.einsum("bkm,bmt->bkt", local_weights, torch.cos(heading))
    resultant = torch.sqrt(sin_heading.square() + cos_heading.square())
    if bool((resultant <= 1e-06).any()):
        raise RediffusionCompatibilityError(
            "barycentric support has an ambiguous circular heading mixture"
        )
    mixed[..., heading_index] = torch.atan2(sin_heading, cos_heading)
    mixed = mixed.to(dtype=bank.dtype)
    if not torch.isfinite(mixed).all():
        raise RediffusionCompatibilityError(
            "barycentric trajectory mixture is non-finite"
        )
    return mixed


def _variance_preserving_noise_mix(
    full_noise_bank: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix original latent slots without collapsing their variance or RMS norm.

    Dividing by ``sqrt(sum(w**2))`` preserves unit variance for independent
    latent slots.  A second deterministic scalar matches the realized RMS to
    the support-weighted RMS energy of those slots.  One-hot support is copied
    back exactly, which makes a proposal slot a strict replay control.
    """
    _validate_trajectory_bank("original DDIM noise bank", full_noise_bank)
    calculation_dtype = (
        torch.float64 if full_noise_bank.dtype == torch.float64 else torch.float32
    )
    noise = full_noise_bank.to(dtype=calculation_dtype)
    local_weights = weights.to(device=noise.device, dtype=calculation_dtype)
    sum_squared = local_weights.square().sum(-1)
    if bool((sum_squared <= 1e-12).any()):
        raise RediffusionCompatibilityError("support weights have zero squared mass")
    variance_scale = sum_squared.rsqrt()
    linear = torch.einsum("bkm,bmtd->bktd", local_weights, noise)
    variance_preserved = linear * variance_scale[..., None, None]
    per_slot_energy = noise.square().mean(dim=(-2, -1))
    target_energy = torch.einsum("bkm,bm->bk", local_weights, per_slot_energy)
    realized_energy = variance_preserved.square().mean(dim=(-2, -1))
    if bool((target_energy <= 0.0).any()) or bool((realized_energy <= 1e-20).any()):
        raise RediffusionCompatibilityError(
            "cannot preserve the norm of a degenerate latent-noise mixture"
        )
    norm_scale = torch.sqrt(target_energy / realized_energy)
    mixed = variance_preserved * norm_scale[..., None, None]
    nonzero = local_weights > 1e-07
    one_hot = nonzero.sum(-1).eq(1) & local_weights.amax(-1).ge(1.0 - 1e-07)
    if bool(one_hot.any()):
        slot = local_weights.argmax(-1)
        exact = _gather_parents(full_noise_bank, slot).to(dtype=calculation_dtype)
        mixed = torch.where(one_hot[..., None, None], exact, mixed)
        variance_scale = torch.where(
            one_hot, torch.ones_like(variance_scale), variance_scale
        )
        norm_scale = torch.where(one_hot, torch.ones_like(norm_scale), norm_scale)
    mixed = mixed.to(dtype=full_noise_bank.dtype)
    if not torch.isfinite(mixed).all():
        raise RediffusionCompatibilityError("mixed DDIM initial noise is non-finite")
    effective_support = sum_squared.reciprocal()
    total_scale = variance_scale * norm_scale
    return (mixed, effective_support, variance_scale, total_scale)


def transfer_final_residual_to_ar(
    ar_proposals: torch.Tensor,
    decoded_proposals: torch.Tensor,
    final_candidates: torch.Tensor,
    parent_indices: torch.Tensor,
    *,
    heading_index: int = 2,
) -> torch.Tensor:
    """Transfer a decoded candidate-parent residual to its matching AR parent.

    Given an original pair ``(ar_parent, decoded_parent)`` and a candidate in
    decoded trajectory space, the returned seed is

    ``ar_parent + (candidate - decoded_parent)``.

    Heading residuals and resulting headings are circularly wrapped.  The
    candidate is converted to the AR bank's device and dtype; the input tensors
    are never mutated.
    """
    _validate_trajectory_bank("ar_proposals", ar_proposals)
    _validate_trajectory_bank("decoded_proposals", decoded_proposals)
    _validate_trajectory_bank("final_candidates", final_candidates)
    if ar_proposals.shape != decoded_proposals.shape:
        raise RediffusionCompatibilityError(
            "ar_proposals and decoded_proposals must have identical shapes"
        )
    if ar_proposals.device != decoded_proposals.device:
        raise RediffusionCompatibilityError(
            "ar_proposals and decoded_proposals must be on the same device"
        )
    if final_candidates.shape[0] != ar_proposals.shape[0]:
        raise RediffusionCompatibilityError(
            "final_candidates batch size does not match the proposal bank"
        )
    if final_candidates.shape[2:] != ar_proposals.shape[2:]:
        raise RediffusionCompatibilityError(
            "final_candidates pose/state dimensions do not match the proposal bank"
        )
    if not 0 <= heading_index < ar_proposals.shape[-1]:
        raise RediffusionCompatibilityError(
            f"heading_index={heading_index} is outside the state dimension"
        )
    if not torch.is_tensor(parent_indices):
        raise RediffusionCompatibilityError("parent_indices must be a torch.Tensor")
    if parent_indices.shape != final_candidates.shape[:2]:
        raise RediffusionCompatibilityError(
            "parent_indices shape must match final_candidates [batch, candidates]"
        )
    ar_parent = _gather_parents(ar_proposals, parent_indices)
    decoded_parent = _gather_parents(decoded_proposals, parent_indices)
    candidate = final_candidates.to(
        device=ar_proposals.device, dtype=ar_proposals.dtype
    )
    residual = candidate - decoded_parent
    residual = residual.clone()
    residual[..., heading_index] = wrap_angle(residual[..., heading_index])
    seed = ar_parent + residual
    seed[..., heading_index] = wrap_angle(seed[..., heading_index])
    if not torch.isfinite(seed).all():
        raise RediffusionCompatibilityError(
            "transferred AR seeds contain non-finite values"
        )
    return seed


@dataclass(frozen=True)
class AlignedARSeedPair:
    """Main/shadow AR seeds aligned by candidate slot and original parent."""

    main: torch.Tensor
    shadow: torch.Tensor
    parent_indices: torch.Tensor


def build_aligned_ar_seed_pair(
    ar_proposals: torch.Tensor,
    decoded_proposals: torch.Tensor,
    main_candidates: torch.Tensor,
    shadow_candidates: torch.Tensor,
    parent_indices: torch.Tensor,
    *,
    heading_index: int = 2,
) -> AlignedARSeedPair:
    """Build main/shadow seeds that use exactly the same original parent slots."""
    _validate_trajectory_bank("main_candidates", main_candidates)
    _validate_trajectory_bank("shadow_candidates", shadow_candidates)
    if main_candidates.shape != shadow_candidates.shape:
        raise RediffusionCompatibilityError(
            "main_candidates and shadow_candidates must have identical shapes"
        )
    main = transfer_final_residual_to_ar(
        ar_proposals,
        decoded_proposals,
        main_candidates,
        parent_indices,
        heading_index=heading_index,
    )
    shadow = transfer_final_residual_to_ar(
        ar_proposals,
        decoded_proposals,
        shadow_candidates,
        parent_indices,
        heading_index=heading_index,
    )
    return AlignedARSeedPair(
        main=main, shadow=shadow, parent_indices=parent_indices.detach().clone()
    )


@dataclass(frozen=True)
class ReplayPreflightReport:
    """Numerical result of replaying the captured original decoder call."""

    exact: bool
    max_abs_error: float
    mean_abs_error: float
    num_values: int


@dataclass(frozen=True)
class RediffusedPair:
    """Decoded main/shadow trajectories produced with aligned noise slots."""

    main: torch.Tensor
    shadow: torch.Tensor
    parent_indices: torch.Tensor


@dataclass(frozen=True)
class LatentFlowTransport:
    """Training-free barycentric latent-flow arms and their provenance.

    ``raw`` is the frozen decoder output for the barycentrically mixed AR seed
    and variance/norm-preserved latent. ``corrected`` transfers the circular
    difference ``raw - linear_final`` onto the direct path.  The caller must
    retain the unmodified direct path as a separate candidate arm.
    """

    raw: torch.Tensor
    corrected: torch.Tensor
    linear_ar: torch.Tensor
    linear_final: torch.Tensor
    mixed_noise: torch.Tensor
    support_weights: torch.Tensor
    effective_support: torch.Tensor
    variance_scale: torch.Tensor
    noise_scale: torch.Tensor
    raw_endpoints: torch.Tensor
    corrected_endpoints: torch.Tensor


class FrozenRediffusionAdapter:
    """Fail-closed adapter around a captured, deterministic DriveSTA DDIM call.

    ``args`` and ``kwargs`` are the values observed by a forward pre-hook on the
    decoder.  The context mapping may be positional (the normal DriveSTA form)
    or passed as the ``context_features`` keyword.  Only ``ar_proposals`` is
    recovered for replay/transport; scene, VLM, ego, and scenario-token
    conditioning are reused by reference.
    """

    def __init__(
        self,
        module: nn.Module,
        args: Sequence[Any],
        kwargs: Optional[Mapping[str, Any]] = None,
        *,
        name: str = "diffusion_decoder",
        require_ddim: bool = True,
    ) -> None:
        if not isinstance(module, nn.Module):
            raise RediffusionCompatibilityError("captured decoder is not an nn.Module")
        self.module = module
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.name = str(name)
        self.require_ddim = bool(require_ddim)
        self._context = self._extract_context()
        self._validate_static_contract()

    @classmethod
    def from_captured(
        cls, captured: Any, *, require_ddim: bool = True
    ) -> "FrozenRediffusionAdapter":
        """Construct from any context object exposing name/module/args/kwargs."""
        missing = [
            attr for attr in ("module", "args", "kwargs") if not hasattr(captured, attr)
        ]
        if missing:
            raise RediffusionCompatibilityError(
                f"captured context is missing attributes: {missing}"
            )
        return cls(
            module=captured.module,
            args=captured.args,
            kwargs=captured.kwargs,
            name=getattr(captured, "name", "diffusion_decoder"),
            require_ddim=require_ddim,
        )

    def _extract_context(self) -> Mapping[str, Any]:
        positional = bool(self.args) and isinstance(self.args[0], Mapping)
        keyword = isinstance(self.kwargs.get("context_features"), Mapping)
        if positional and keyword:
            raise RediffusionCompatibilityError(
                "captured decoder call contains two context_features mappings"
            )
        if positional:
            return self.args[0]
        if keyword:
            return self.kwargs["context_features"]
        raise RediffusionCompatibilityError(
            "captured decoder call has no positional or keyword context_features mapping"
        )

    def _validate_static_contract(self) -> None:
        if "ar_proposals" not in self._context:
            raise RediffusionCompatibilityError(
                "captured context_features has no `ar_proposals`"
            )
        _validate_trajectory_bank(
            "captured ar_proposals", self._context["ar_proposals"]
        )
        if self.module.training:
            raise RediffusionCompatibilityError(
                "captured diffusion decoder is in training mode; refusing test-time replay"
            )
        if self.require_ddim:
            diffusion_type = getattr(self.module, "_diffusion_type", None)
            if str(diffusion_type).lower() != "ddim":
                raise RediffusionCompatibilityError(
                    f"{self.name!r} is not a recognized DDIM decoder"
                )
        if getattr(self.module, "noise_scheduler", None) is None:
            raise RediffusionCompatibilityError(
                f"{self.name!r} has no initialized DDIM noise_scheduler"
            )
        for method_name in ("_sample_initial_noise", "_run_dit", "_decode_sample"):
            if not callable(getattr(self.module, method_name, None)):
                raise RediffusionCompatibilityError(
                    f"{self.name!r} has no compatible {method_name} method"
                )
        inference_steps = getattr(self.module, "_num_inference_steps", None)
        if not isinstance(inference_steps, int) or inference_steps <= 0:
            raise RediffusionCompatibilityError(
                f"{self.name!r} has no valid DDIM inference-step count"
            )

    @property
    def captured_ar_proposals(self) -> torch.Tensor:
        return self._context["ar_proposals"]

    def _validate_deterministic_noise_contract(self, seeds: torch.Tensor) -> None:
        if getattr(self.module, "_eval_noise_seed", None) is None:
            raise RediffusionCompatibilityError(
                "decoder has no fixed _eval_noise_seed; aligned pair noise is not guaranteed"
            )
        scenario_tokens = self._context.get("scenario_token")
        if scenario_tokens is None:
            raise RediffusionCompatibilityError(
                "captured context has no scenario_token; deterministic noise is not guaranteed"
            )
        try:
            token_count = len(scenario_tokens)
        except TypeError as exc:
            raise RediffusionCompatibilityError(
                "scenario_token must be a batch-sized sequence"
            ) from exc
        if token_count != seeds.shape[0]:
            raise RediffusionCompatibilityError(
                f"scenario_token batch size does not match AR seeds: {token_count} != {seeds.shape[0]}"
            )

    def _prepared_seeds(self, seeds: torch.Tensor) -> torch.Tensor:
        _validate_trajectory_bank("rediffusion AR seeds", seeds)
        captured = self.captured_ar_proposals
        if seeds.shape[0] != captured.shape[0] or seeds.shape[2:] != captured.shape[2:]:
            raise RediffusionCompatibilityError(
                "rediffusion AR seeds must match captured batch/pose/state dimensions"
            )
        return seeds.to(device=captured.device, dtype=captured.dtype)

    def _sample_noise(self, proposals: torch.Tensor) -> torch.Tensor:
        self._validate_deterministic_noise_contract(proposals)
        try:
            with torch.no_grad():
                noise = self.module._sample_initial_noise(
                    proposals.shape[0],
                    proposals.shape[1],
                    proposals,
                    self._context.get("scenario_token"),
                )
        except Exception as exc:
            raise RediffusionCompatibilityError(
                f"failed to recreate deterministic noise through {self.name!r}"
            ) from exc
        if not torch.is_tensor(noise) or noise.shape != proposals.shape:
            shape = (
                tuple(noise.shape) if torch.is_tensor(noise) else type(noise).__name__
            )
            raise RediffusionCompatibilityError(
                f"decoder initial noise has invalid shape/type: {shape}"
            )
        noise = noise.to(device=proposals.device, dtype=proposals.dtype).detach()
        if not torch.isfinite(noise).all():
            raise RediffusionCompatibilityError("decoder initial noise is non-finite")
        return noise

    def _original_parent_noise(self, parent_slots: torch.Tensor) -> torch.Tensor:
        """Recreate the full original noise bank and gather arbitrary parents."""
        original_ar = self.captured_ar_proposals
        original_noise = self._sample_noise(original_ar)
        return _gather_parents(original_noise, parent_slots)

    def _run_ddim(
        self, ar_seeds: torch.Tensor, *, initial_noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run the captured DDIM primitives with an optional explicit x_T."""
        if self.module.training:
            raise RediffusionCompatibilityError(
                "diffusion decoder entered training mode after capture"
            )
        seeds = self._prepared_seeds(ar_seeds)
        if initial_noise is None:
            x = self._sample_noise(seeds)
        else:
            _validate_trajectory_bank("explicit DDIM initial_noise", initial_noise)
            if initial_noise.shape != seeds.shape:
                raise RediffusionCompatibilityError(
                    "explicit DDIM initial_noise shape must equal the AR seed shape"
                )
            x = initial_noise.to(device=seeds.device, dtype=seeds.dtype).detach()
        scheduler = self.module.noise_scheduler
        try:
            with torch.no_grad():
                scheduler.set_timesteps(self.module._num_inference_steps, seeds.device)
                for timestep_scalar in scheduler.timesteps:
                    timestep = torch.as_tensor(timestep_scalar, device=seeds.device)
                    timesteps = timestep.expand(seeds.shape[0])
                    pred_noise = self.module._run_dit(
                        x,
                        seeds,
                        timesteps,
                        self._context.get("vlm_hidden_state"),
                        self._context.get("ego_status_feature"),
                    )
                    if not torch.is_tensor(pred_noise) or pred_noise.shape != x.shape:
                        raise RediffusionCompatibilityError(
                            "decoder _run_dit returned an invalid noise prediction"
                        )
                    if not torch.isfinite(pred_noise).all():
                        raise RediffusionCompatibilityError(
                            "decoder _run_dit returned non-finite values"
                        )
                    step = scheduler.step(
                        model_output=pred_noise, timestep=timestep_scalar, sample=x
                    )
                    x = getattr(step, "prev_sample", None)
                    if not torch.is_tensor(x) or x.shape != seeds.shape:
                        raise RediffusionCompatibilityError(
                            "DDIM scheduler step returned an invalid prev_sample"
                        )
                proposals = self.module._decode_sample(seeds, x)
        except RediffusionCompatibilityError:
            raise
        except Exception as exc:
            raise RediffusionCompatibilityError(
                f"frozen DDIM loop through {self.name!r} failed"
            ) from exc
        if not torch.is_tensor(proposals) or proposals.shape != seeds.shape:
            raise RediffusionCompatibilityError(
                "decoder _decode_sample returned an invalid proposal tensor"
            )
        proposals = proposals.detach()
        if not torch.isfinite(proposals).all():
            raise RediffusionCompatibilityError(
                "diffusion decoder returned non-finite proposals"
            )
        return proposals

    def rediffuse(self, ar_seeds: torch.Tensor) -> torch.Tensor:
        """Replay a full bank whose slots match the captured original bank.

        Subsets or reordered candidates must use :meth:`rediffuse_pair`, which
        gathers their original parent-slot noise explicitly.
        """
        if ar_seeds.shape != self.captured_ar_proposals.shape:
            raise RediffusionCompatibilityError(
                "rediffuse only accepts a full slot-aligned AR bank; use rediffuse_pair for candidate subsets"
            )
        return self._run_ddim(ar_seeds)

    def replay_preflight(
        self,
        expected_proposals: torch.Tensor,
        *,
        atol: Optional[float] = None,
        rtol: float = 1e-05,
    ) -> ReplayPreflightReport:
        """Replay the original AR bank and require agreement with normal output."""
        if atol is None:
            atol = (
                1e-05
                if self.captured_ar_proposals.dtype in (torch.float32, torch.float64)
                else 0.002
            )
        if atol < 0.0 or rtol < 0.0 or (not math.isfinite(atol + rtol)):
            raise ValueError("preflight tolerances must be finite and non-negative")
        replay = self.rediffuse(self.captured_ar_proposals)
        if not torch.is_tensor(expected_proposals):
            raise RediffusionCompatibilityError("expected_proposals must be a tensor")
        expected = expected_proposals.to(device=replay.device, dtype=replay.dtype)
        if expected.shape != replay.shape:
            raise RediffusionCompatibilityError(
                f"normal decoded proposals do not match replay shape: {tuple(expected.shape)} != {tuple(replay.shape)}"
            )
        if not torch.isfinite(expected).all():
            raise RediffusionCompatibilityError(
                "normal decoded proposals contain non-finite values"
            )
        error = (replay - expected).abs()
        report = ReplayPreflightReport(
            exact=bool(torch.equal(replay, expected)),
            max_abs_error=float(error.max().item()) if error.numel() else 0.0,
            mean_abs_error=float(error.mean().item()) if error.numel() else 0.0,
            num_values=error.numel(),
        )
        if not torch.allclose(replay, expected, atol=atol, rtol=rtol):
            raise RediffusionCompatibilityError(
                f"captured DDIM replay does not reproduce normal proposals: max_abs_error={report.max_abs_error:.6g}, mean_abs_error={report.mean_abs_error:.6g}"
            )
        return report

    @torch.no_grad()
    def latent_flow_transport(
        self,
        direct_candidates: torch.Tensor,
        decoded_proposal_pool: torch.Tensor,
        support_weights: torch.Tensor,
        *,
        heading_index: int = 2,
    ) -> LatentFlowTransport:
        """Decode local barycentric latent arms with the frozen Stage-3 DDIM.

        Args:
            direct_candidates: Complete particle paths in final trajectory
                space, shaped ``[B,K,T,D]``.
            decoded_proposal_pool: The exact normal Stage-3 decoded proposal
                bank captured by the scorer, shaped ``[B,M,T,D]``.
            support_weights: Each particle's barycentric provenance over the
                *original* proposal slots, shaped ``[B,K,M]``. Rows are
                normalized internally and must be finite and non-negative.
            heading_index: Circular heading coordinate in the final state.

        This operation intentionally supports only the released decoder form
        with ``diffusion_use_ar_proposal_tokens=false``. If proposals condition
        the denoiser, mixing both the proposal and latent changes the semantics
        and requires a separately validated method.
        """
        if getattr(self.module, "_use_ar_proposal_tokens", None) is not False:
            raise RediffusionCompatibilityError(
                "latent-flow transport requires diffusion_use_ar_proposal_tokens=false"
            )
        _validate_trajectory_bank("direct_candidates", direct_candidates)
        _validate_trajectory_bank("decoded_proposal_pool", decoded_proposal_pool)
        original_ar = self.captured_ar_proposals
        if decoded_proposal_pool.shape != original_ar.shape:
            raise RediffusionCompatibilityError(
                "decoded_proposal_pool must match the captured original AR bank"
            )
        if decoded_proposal_pool.device != original_ar.device:
            raise RediffusionCompatibilityError(
                "decoded and AR proposal banks must be on the same device"
            )
        if (
            direct_candidates.shape[0] != original_ar.shape[0]
            or direct_candidates.shape[2:] != original_ar.shape[2:]
        ):
            raise RediffusionCompatibilityError(
                "direct candidate batch/pose/state dimensions must match the proposal bank"
            )
        weights = _normalize_support_weights(support_weights, original_ar)
        if weights.shape[1] != direct_candidates.shape[1]:
            raise RediffusionCompatibilityError(
                "support_weights candidate dimension does not match direct_candidates"
            )
        linear_ar = _barycentric_trajectory_mix(
            original_ar, weights, heading_index=heading_index
        )
        linear_final = _barycentric_trajectory_mix(
            decoded_proposal_pool, weights, heading_index=heading_index
        )
        full_noise = self._sample_noise(original_ar)
        mixed_noise, effective_support, variance_scale, noise_scale = (
            _variance_preserving_noise_mix(full_noise, weights)
        )
        raw = self._run_ddim(linear_ar, initial_noise=mixed_noise)
        direct = direct_candidates.to(device=raw.device, dtype=raw.dtype)
        correction = raw - linear_final.to(device=raw.device, dtype=raw.dtype)
        correction = correction.clone()
        correction[..., heading_index] = wrap_angle(correction[..., heading_index])
        corrected = direct + correction
        corrected[..., heading_index] = wrap_angle(corrected[..., heading_index])
        corrected = corrected.detach()
        if not torch.isfinite(raw).all() or not torch.isfinite(corrected).all():
            raise RediffusionCompatibilityError(
                "latent-flow transport produced non-finite trajectories"
            )
        return LatentFlowTransport(
            raw=raw.detach(),
            corrected=corrected,
            linear_ar=linear_ar.detach(),
            linear_final=linear_final.detach(),
            mixed_noise=mixed_noise.detach(),
            support_weights=weights.detach(),
            effective_support=effective_support.detach(),
            variance_scale=variance_scale.detach(),
            noise_scale=noise_scale.detach(),
            raw_endpoints=raw[..., -1, :].detach(),
            corrected_endpoints=corrected[..., -1, :].detach(),
        )

    def rediffuse_aligned_pair(self, pair: AlignedARSeedPair) -> RediffusedPair:
        """Re-diffuse main and shadow in aligned calls with shared DDIM noise.

        The full original-mode deterministic noise bank is recreated and
        gathered using ``parent_indices``. Main and shadow then receive that
        same gathered tensor in two aligned DDIM loops. A single shared
        ``parent_indices`` tensor records that both columns originate from the
        same original AR parent.
        """
        if pair.main.shape != pair.shadow.shape:
            raise RediffusionCompatibilityError(
                "aligned main/shadow AR seed shapes differ"
            )
        if pair.parent_indices.shape != pair.main.shape[:2]:
            raise RediffusionCompatibilityError(
                "aligned parent_indices do not match AR seed candidate slots"
            )
        self._validate_deterministic_noise_contract(pair.main)
        parent_noise = self._original_parent_noise(pair.parent_indices)
        main = self._run_ddim(pair.main, initial_noise=parent_noise)
        shadow = self._run_ddim(pair.shadow, initial_noise=parent_noise)
        return RediffusedPair(
            main=main,
            shadow=shadow,
            parent_indices=pair.parent_indices.detach().clone(),
        )

    def rediffuse_pair(
        self,
        direct_candidates: torch.Tensor,
        decoded_parent_pool: torch.Tensor,
        parent_slots: torch.Tensor,
        *,
        heading_index: int = 2,
        shadow_atol: Optional[float] = None,
        shadow_rtol: float = 1e-05,
    ) -> RediffusedPair:
        """Re-diffuse candidates beside their same-noise original-parent controls.

        This is the high-level runtime entry point. ``direct_candidates`` are
        candidate trajectories in the final/decoded space;
        ``decoded_parent_pool`` is the matching normal Stage3 final proposal
        bank; and ``parent_slots[b, k]`` identifies the original parent of
        candidate ``k``.  The captured context supplies the corresponding AR
        proposal bank.

        The main seed receives the candidate-to-final-parent residual.  The
        shadow candidate is the final parent itself, so its transferred seed is
        exactly the original AR parent.  Re-diffusing both in aligned calls
        gives a paired control under the same deterministic DDIM noise slot.
        """
        _validate_trajectory_bank("decoded_parent_pool", decoded_parent_pool)
        _validate_trajectory_bank("direct_candidates", direct_candidates)
        if decoded_parent_pool.shape != self.captured_ar_proposals.shape:
            raise RediffusionCompatibilityError(
                "decoded_parent_pool must match the captured AR proposal-bank shape"
            )
        if not torch.is_tensor(parent_slots):
            raise RediffusionCompatibilityError("parent_slots must be a torch.Tensor")
        if parent_slots.shape != direct_candidates.shape[:2]:
            raise RediffusionCompatibilityError(
                "parent_slots shape must match direct_candidates [batch, candidates]"
            )
        if shadow_atol is None:
            shadow_atol = (
                1e-05
                if decoded_parent_pool.dtype in (torch.float32, torch.float64)
                else 0.002
            )
        if (
            shadow_atol < 0.0
            or shadow_rtol < 0.0
            or (not math.isfinite(shadow_atol + shadow_rtol))
        ):
            raise ValueError("shadow replay tolerances must be finite and non-negative")
        shadow_candidates = _gather_parents(decoded_parent_pool, parent_slots)
        seed_pair = build_aligned_ar_seed_pair(
            self.captured_ar_proposals,
            decoded_parent_pool,
            direct_candidates,
            shadow_candidates,
            parent_slots,
            heading_index=heading_index,
        )
        output = self.rediffuse_aligned_pair(seed_pair)
        expected_shadow = shadow_candidates.to(
            device=output.shadow.device, dtype=output.shadow.dtype
        )
        if not torch.allclose(
            output.shadow, expected_shadow, atol=shadow_atol, rtol=shadow_rtol
        ):
            error = (output.shadow - expected_shadow).abs()
            raise RediffusionCompatibilityError(
                f"parent-slot shadow replay does not reproduce the original decoded parent: max_abs_error={float(error.max().item()):.6g}"
            )
        return output
