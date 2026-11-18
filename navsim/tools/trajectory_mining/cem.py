"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
from typing import Sequence
import torch


def make_sample_generators(
    device: torch.device, seeds: Sequence[int]
) -> list[torch.Generator]:
    """Create private per-sample RNGs without touching PyTorch's global RNG."""
    generators: list[torch.Generator] = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)
    return generators
