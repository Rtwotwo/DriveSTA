"""Extracted from the existing Endpoint implementation; inference only."""

from __future__ import annotations
import torch
from navsim.tools.trajectory_mining.config import TTOConfig


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _terminal_speed(proposals: torch.Tensor, dt: float) -> torch.Tensor:
    return (
        torch.linalg.vector_norm(
            proposals[..., -1, :2] - proposals[..., -2, :2], dim=-1
        )
        / dt
    )


def _path_distance(proposals: torch.Tensor, config: TTOConfig) -> torch.Tensor:
    """Compatibility includes the whole timed path, not just its endpoint."""
    ade = torch.linalg.vector_norm(
        proposals[:, None, :, :2] - proposals[None, :, :, :2], dim=-1
    ).mean(-1)
    heading = _wrap(proposals[:, None, :, 2] - proposals[None, :, :, 2]).abs().amax(-1)
    speed = _terminal_speed(proposals, config.trajectory_dt)
    speed_difference = (speed[:, None] - speed[None, :]).abs()
    return torch.maximum(
        torch.maximum(
            ade / config.support_max_ade, heading / config.support_max_heading
        ),
        speed_difference / config.endpoint_mode_speed_tolerance,
    )


def _neighborhoods(
    proposals: torch.Tensor,
    scores: torch.Tensor,
    finite: torch.Tensor,
    config: TTOConfig,
) -> list[torch.Tensor]:
    """Farthest-first local supports, starting with the best original score.

    Every accepted member is compatible with *all* previous members, avoiding
    a star neighborhood which accidentally mixes two incompatible branches.
    These are local supports, not a semantic route or obstacle classifier.
    """
    if not bool(finite.any()):
        return []
    distance = _path_distance(proposals, config)
    centers: list[int] = []
    supports: list[torch.Tensor] = []
    center = int(scores.masked_fill(~finite, -torch.inf).argmax())
    min_distance = torch.full_like(scores, torch.inf)
    for _ in range(config.endpoint_modes):
        centers.append(center)
        accepted = [center]
        for candidate in torch.argsort(distance[center], stable=True).tolist():
            if candidate == center or not bool(finite[candidate]):
                continue
            if bool((distance[candidate, accepted] <= 1.0).all()):
                accepted.append(candidate)
            if len(accepted) >= config.support_neighbors:
                break
        supports.append(
            torch.tensor(accepted, dtype=torch.long, device=proposals.device)
        )
        min_distance = torch.minimum(min_distance, distance[center])
        remaining = min_distance.masked_fill(~finite, -torch.inf)
        remaining[centers] = -torch.inf
        if not bool(torch.isfinite(remaining).any()) or float(remaining.max()) <= 1e-05:
            break
        center = int(remaining.argmax())
    return supports


def _surface(
    queries: torch.Tensor,
    observations: torch.Tensor,
    scores: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    difference = queries[:, None] - observations[None]
    mahalanobis = torch.einsum(
        "qnd,df,qnf->qn", difference, torch.linalg.inv(covariance), difference
    )
    kernel = torch.softmax(-0.5 * mahalanobis, dim=-1)
    return kernel @ scores


def _mixture_trials(
    endpoints: torch.Tensor,
    center: torch.Tensor,
    radius: float,
    scores: torch.Tensor,
    count: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sparse convex mixtures; no coordinate-wise box or extrapolation."""
    members = endpoints.shape[0]
    random = torch.rand(
        (count, members), generator=generator, device=endpoints.device
    ).clamp_min(1e-07)
    sparse = torch.zeros_like(random)
    selected = torch.topk(random, min(3, members), dim=-1).indices
    mass = -torch.log(
        torch.rand(
            (count, selected.shape[1]), generator=generator, device=endpoints.device
        ).clamp_min(1e-07)
    )
    sparse.scatter_(1, selected, mass)
    sparse = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-08)
    sparse[0] = torch.softmax((scores - scores.max()) / temperature, dim=-1)
    return (1.0 - radius) * center[None] + radius * sparse
