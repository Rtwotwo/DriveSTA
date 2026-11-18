"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Tuple, Dict
import torch
import torch.nn as nn


def _expand_batch_value(
    value: Any, sample_indices: torch.Tensor, source_batch: int
) -> Any:
    """Select batch-aligned tensors while preserving nested call structures."""
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == source_batch:
            return value.index_select(0, sample_indices.to(device=value.device))
        return value
    if isinstance(value, Mapping):
        items = [
            (key, _expand_batch_value(item, sample_indices, source_batch))
            for key, item in value.items()
        ]
        if type(value) is dict:
            return dict(items)
        try:
            return type(value)(items)
        except (TypeError, ValueError):
            return dict(items)
    if isinstance(value, list):
        return [
            _expand_batch_value(item, sample_indices, source_batch) for item in value
        ]
    if isinstance(value, tuple):
        items = tuple(
            (_expand_batch_value(item, sample_indices, source_batch) for item in value)
        )
        if hasattr(value, "_fields"):
            return type(value)(*items)
        if type(value) is tuple:
            return items
        try:
            return type(value)(items)
        except TypeError:
            return items
    return value


def _replacement_call(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    proposals: torch.Tensor,
    sample_indices: torch.Tensor,
    source_batch: int,
) -> Any:
    """Replay a captured scorer call with a replacement proposal batch."""
    expanded_kwargs = {
        key: (
            value
            if key == "proposals"
            else _expand_batch_value(value, sample_indices, source_batch)
        )
        for key, value in kwargs.items()
    }
    expanded_args = list(_expand_batch_value(tuple(args), sample_indices, source_batch))
    if "proposals" in expanded_kwargs:
        expanded_kwargs["proposals"] = proposals
    elif expanded_args:
        expanded_args[0] = proposals
    else:
        raise RuntimeError(
            "Captured scorer call has no `proposals` keyword or positional argument"
        )
    return module(*expanded_args, **expanded_kwargs)


def _component_probabilities(
    output: Any, expected_shape: Tuple[int, int]
) -> Dict[str, torch.Tensor]:
    """Return finite FP32 sigmoid probabilities from DriveSTA score logits."""
    if not isinstance(output, Mapping) or "pred_logit" not in output:
        return {}
    pred_logit = output["pred_logit"]
    if pred_logit is None:
        return {}
    if not isinstance(pred_logit, Mapping):
        raise TypeError("scorer output `pred_logit` must be a mapping")
    components: Dict[str, torch.Tensor] = {}
    for name, logits in pred_logit.items():
        if not isinstance(name, str):
            raise TypeError("scorer component names must be strings")
        if not torch.is_tensor(logits):
            raise TypeError(f"scorer component {name!r} must be a torch.Tensor")
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"scorer component {name!r} returned shape {tuple(logits.shape)}; expected {expected_shape}"
            )
        logits_fp32 = logits.float()
        if not torch.isfinite(logits_fp32).all():
            raise FloatingPointError(f"scorer component {name!r} contains NaN or Inf")
        probability = torch.sigmoid(logits_fp32)
        if not torch.isfinite(probability).all():
            raise FloatingPointError(
                f"scorer component {name!r} produced non-finite probabilities"
            )
        components[name] = probability
    return components
