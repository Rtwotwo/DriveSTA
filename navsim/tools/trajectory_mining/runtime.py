"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
import hashlib
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from navsim.tools.trajectory_mining.config import TTOConfig
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class _ScorerContext:
    name: str
    module: nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class _GeneratorContext:
    """Inputs captured from the already-loaded frozen trajectory decoder."""

    name: str
    module: nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class DriveSTATestTimeOptimizer:

    def __init__(self, config: TTOConfig | Mapping[str, Any] | Any = None):
        self.config = TTOConfig.from_mapping(config)
        self._captured_calls: list[_ScorerContext] = []
        self._captured_generator_calls: list[_GeneratorContext] = []
        self._batch_count = 0
        self._postprocessor_paths: list[str] = []
        self._diagnostic_path: Optional[Path] = None
        if self.enabled:
            logger.info(
                "DriveSTA-TTO effective configuration: %s",
                json.dumps(self.config.to_dict(), sort_keys=True),
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @classmethod
    def from_mapping(cls, value: Any) -> "DriveSTATestTimeOptimizer":
        return cls(TTOConfig.from_mapping(value))

    def _candidate_scorers(self, agent: nn.Module) -> list[tuple[str, nn.Module]]:
        modules = dict(agent.named_modules())
        explicit = self.config.scorer_module_path
        if explicit:
            if explicit not in modules:
                choices = sorted(
                    (name for name in modules if name.endswith("score_decoder"))
                )
                raise RuntimeError(
                    f"scorer_module_path={explicit!r} was not found; candidates={choices}"
                )
            return [(explicit, modules[explicit])]
        candidates = [
            (name, module)
            for name, module in modules.items()
            if name.endswith("score_decoder")
            and callable(getattr(module, "forward", None))
            and hasattr(module, "pos_embed")
        ]
        if not candidates:
            raise RuntimeError(
                "No DriveSTA score_decoder was found. Set test_time_optimizer.scorer_module_path for a custom model."
            )
        return candidates

    def _candidate_generators(self, agent: nn.Module) -> list[tuple[str, nn.Module]]:
        """Locate the frozen Stage3 diffusion decoder for goal-particle routes.

        Historical strategies never install this hook.  The structural checks
        intentionally match the deployed decoder contract instead of a class
        import, keeping the package independent of the host NAVSIM checkout.
        """
        modules = dict(agent.named_modules())
        explicit = self.config.generator_module_path
        if explicit:
            if explicit not in modules:
                choices = sorted(
                    (name for name in modules if name.endswith("diffusion_decoder"))
                )
                raise RuntimeError(
                    f"generator_module_path={explicit!r} was not found; candidates={choices}"
                )
            candidates = [(explicit, modules[explicit])]
        else:
            candidates = [
                (name, module)
                for name, module in modules.items()
                if name.endswith("diffusion_decoder")
                and callable(getattr(module, "forward", None))
            ]
        compatible = [
            (name, module)
            for name, module in candidates
            if callable(getattr(module, "_sample_initial_noise", None))
            and callable(getattr(module, "_run_dit", None))
            and callable(getattr(module, "_decode_sample", None))
        ]
        if not compatible:
            raise RuntimeError(
                "goal-particle TTO requires the loaded frozen Stage3 diffusion_decoder; set test_time_optimizer.generator_module_path when the module uses a custom name"
            )
        return compatible

    @contextmanager
    def capture_scorer(self, agent: nn.Module) -> Iterator[None]:
        """Capture scorer context around exactly one ordinary agent forward."""
        if not self.enabled:
            yield
            return
        self._captured_calls = []
        self._captured_generator_calls = []
        self._postprocessor_paths = [
            name for name, _ in agent.named_modules() if name.endswith("safety_gate")
        ]

        def make_hook(name: str):

            def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
                self._captured_calls.append(
                    _ScorerContext(name, module, tuple(args), dict(kwargs))
                )

            return hook

        def make_generator_hook(name: str):

            def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
                self._captured_generator_calls.append(
                    _GeneratorContext(name, module, tuple(args), dict(kwargs))
                )

            return hook

        try:
            with ExitStack() as stack:
                for name, module in self._candidate_scorers(agent):
                    handle = module.register_forward_pre_hook(
                        make_hook(name), with_kwargs=True
                    )
                    stack.callback(handle.remove)
                needs_goal_generator = (
                    self.config.strategy == "goal_particle_rediffusion"
                    and self.config.goal_particle_rediffusion
                    or (
                        self.config.strategy == "goal_particle_selective"
                        and self.config.goal_particle_rediffusion
                        and self.config.goal_particle_selective_include_corrected
                    )
                )
                if needs_goal_generator:
                    for name, module in self._candidate_generators(agent):
                        handle = module.register_forward_pre_hook(
                            make_generator_hook(name), with_kwargs=True
                        )
                        stack.callback(handle.remove)
                yield
        except BaseException:
            self._captured_calls = []
            self._captured_generator_calls = []
            self._postprocessor_paths = []
            raise

    def _captured_context(self) -> _ScorerContext:
        if not self._captured_calls:
            raise RuntimeError(
                "The DriveSTA scorer was not called during agent.forward; cannot run test-time optimization."
            )
        names = {call.name for call in self._captured_calls}
        if len(names) > 1:
            raise RuntimeError(
                f"Multiple score decoders ran during forward: {sorted(names)}. Set scorer_module_path explicitly."
            )
        return self._captured_calls[-1]

    def _captured_generator_context(self) -> _GeneratorContext:
        if not self._captured_generator_calls:
            raise RuntimeError(
                "The Stage3 diffusion decoder was not called during agent.forward; cannot run goal-conditioned frozen re-diffusion."
            )
        names = {call.name for call in self._captured_generator_calls}
        if len(names) > 1:
            raise RuntimeError(
                f"Multiple diffusion decoders ran during forward: {sorted(names)}. Set generator_module_path explicitly."
            )
        return self._captured_generator_calls[-1]

    @staticmethod
    def _invoke_with_proposals(context: _ScorerContext, proposals: torch.Tensor) -> Any:
        args = list(context.args)
        kwargs = dict(context.kwargs)
        if "proposals" in kwargs:
            kwargs["proposals"] = proposals
        elif args:
            args[0] = proposals
        else:
            raise RuntimeError(
                f"Captured scorer {context.name!r} has no proposals argument"
            )
        return context.module(*args, **kwargs)

    @staticmethod
    def _stable_drivesta_score(module: nn.Module, output: Any) -> torch.Tensor:
        """Aggregate scorer logits in FP32, including older DriveSTA variants."""
        if isinstance(output, Mapping) and torch.is_tensor(output.get("pdm_score")):
            pdm_score = output["pdm_score"].float()
            if torch.isfinite(pdm_score).all():
                return pdm_score
        if isinstance(output, Mapping) and isinstance(
            output.get("pred_logit"), Mapping
        ):
            logits = output["pred_logit"]
            required = {
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "driving_direction_compliance",
                "time_to_collision_within_bound",
                "ego_progress",
                "comfort",
            }
            if required.issubset(logits):
                reference = logits["no_at_fault_collisions"]
                score = torch.zeros_like(reference, dtype=torch.float32)
                for attr, key in (
                    ("noc_weight", "no_at_fault_collisions"),
                    ("dac_weight", "drivable_area_compliance"),
                    ("ddc_weight", "driving_direction_compliance"),
                ):
                    weight = float(getattr(module, attr))
                    if weight > 0:
                        score = score + weight * F.logsigmoid(logits[key].float())
                quality = []
                for attr, key in (
                    ("ttc_weight", "time_to_collision_within_bound"),
                    ("ep_weight", "ego_progress"),
                    ("comfort_weight", "comfort"),
                ):
                    weight = float(getattr(module, attr))
                    if weight > 0:
                        quality.append(
                            F.logsigmoid(logits[key].float())
                            + torch.log(torch.tensor(weight, device=reference.device))
                        )
                if quality:
                    return score + torch.logsumexp(torch.stack(quality), dim=0)
        if isinstance(output, Mapping) and torch.is_tensor(output.get("pdm_score")):
            return output["pdm_score"].float()
        if torch.is_tensor(output):
            return output.float()
        raise RuntimeError("Score decoder output has neither pred_logit nor pdm_score")

    @staticmethod
    def _repeat_reference(reference: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return reference[:, :0]
        available = reference.shape[1]
        if available < 1:
            raise ValueError("at least one reference proposal is required")
        repeats = (count + available - 1) // available
        return reference.repeat(1, repeats, 1, 1)[:, :count]

    def _score_fn(self, context: _ScorerContext, reference_proposals: torch.Tensor):

        def score(trajectories: torch.Tensor) -> torch.Tensor:
            query_count = trajectories.shape[1]
            expected_count = int(
                getattr(context.module, "proposal_num", reference_proposals.shape[1])
            )
            if query_count > expected_count:
                raise ValueError(
                    f"CEM requested {query_count} candidates but the scorer was trained with proposal_num={expected_count}. Chunking is intentionally disabled because DriveSTA scorer self-attention makes scores set-dependent."
                )
            if query_count < expected_count:
                padding = self._repeat_reference(
                    reference_proposals, expected_count - query_count
                )
                scorer_trajectories = torch.cat((trajectories, padding), dim=1)
            else:
                scorer_trajectories = trajectories
            parameter = next(context.module.parameters(), None)
            scorer_dtype = (
                parameter.dtype if parameter is not None else scorer_trajectories.dtype
            )
            scorer_input = scorer_trajectories.to(dtype=scorer_dtype)
            output = self._invoke_with_proposals(context, scorer_input)
            result = self._stable_drivesta_score(context.module, output)
            expected_shape = scorer_trajectories.shape[:2]
            if result.shape != expected_shape:
                raise RuntimeError(
                    f"DriveSTA scorer returned shape {tuple(result.shape)} for trajectories {tuple(scorer_trajectories.shape)}"
                )
            return result[:, :query_count].to(
                device=trajectories.device, dtype=torch.float32
            )

        return score

    def _sample_seeds(
        self, sample_ids: Optional[Sequence[Any]], batch: int
    ) -> list[int]:
        if sample_ids is None:
            sample_ids = tuple(range(batch))
        if len(sample_ids) != batch:
            raise ValueError(f"expected {batch} sample ids, got {len(sample_ids)}")
        seeds = []
        for sample_id in sample_ids:
            payload = f"{self.config.seed}:{sample_id}".encode("utf-8")
            digest = hashlib.blake2b(payload, digest_size=8).digest()
            seeds.append(int.from_bytes(digest, "little") & (1 << 63) - 1)
        return seeds

    @torch.no_grad()
    def refine_predictions(
        self,
        predictions: MutableMapping[str, Any],
        features: Mapping[str, Any],
        *,
        sample_ids: Optional[Sequence[Any]] = None,
    ) -> MutableMapping[str, Any]:
        """Refine ``predictions[trajectory_key]`` in place when enabled."""
        if not self.enabled:
            return predictions
        original_trajectory = predictions.get(self.config.trajectory_key)
        original_diagnostics = {
            key: value for key, value in predictions.items() if key.startswith("tto_")
        }
        try:
            return self._refine_impl(predictions, features, sample_ids)
        except Exception:
            if self.config.on_error == "raise":
                raise
            if original_trajectory is not None:
                predictions[self.config.trajectory_key] = original_trajectory
            for key in tuple(predictions):
                if key.startswith("tto_"):
                    predictions.pop(key)
            predictions.update(original_diagnostics)
            logger.exception("TOAD-TTO failed; keeping the original trajectory")
            return predictions
        finally:
            self._captured_calls = []
            self._captured_generator_calls = []

    def _refine_impl(self, predictions, features, sample_ids):
        from navsim.tools.trajectory_mining.goal_particle_selective_runtime import refine_goal_particle_selective_predictions

        return refine_goal_particle_selective_predictions(
            self, predictions, features, sample_ids
        )
