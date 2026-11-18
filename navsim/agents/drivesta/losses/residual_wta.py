"""Per-parent local dense WTA. Target, noise label and winner stay paired."""

import torch
from navsim.agents.drivesta.losses.refinement import trajectory_distance


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.no_grad()
def make_pairs(parents, targets, retained, config):
    bank = targets["auxiliary_trajectories"].float()
    mask = targets["auxiliary_mask"].bool()
    labels = targets["auxiliary_labels"].float()
    if bank.ndim != 4 or bank.shape[2:] != (8, 3):
        raise ValueError("Expected screened trajectory bank [B,K,8,3]")
    admitted = (labels[..., 6] >= 0.95) & (labels[..., [0, 1, 3, 4, 5]] == 1).all(-1)
    if (mask & ~admitted).any():
        raise ValueError("Generator bank contains unqualified official labels")
    difference = parents[:, :, None].float() - bank[:, None]
    ade = difference[..., :2].norm(dim=-1).mean(-1)
    fde = difference[..., -1, :2].norm(dim=-1)
    heading = wrap(difference[..., 2]).abs().mean(-1)
    eligible = (
        mask[:, None]
        & (ade <= config.pair_max_ade)
        & (fde <= config.pair_max_fde)
        & (heading <= config.pair_max_heading)
    )
    cost = trajectory_distance(parents[:, :, None], bank[:, None])
    k = min(int(config.wta_targets_per_parent), bank.shape[1])
    if k < 1:
        raise ValueError("Target capacity must be positive; empty scenes use mask")
    idx = cost.masked_fill(~eligible, torch.inf).topk(k, dim=-1, largest=False).indices
    b = torch.arange(len(parents), device=parents.device)[:, None, None]
    selected = bank[b, idx].clone()
    valid = eligible.gather(-1, idx)
    matched = valid.any(-1)
    # Unmatched does NOT mean safe. Do not force a remote driving intent.
    selected = torch.where(
        matched[..., None, None, None], selected, parents[:, :, None]
    )
    valid[..., 0] |= ~matched
    active = torch.ones_like(matched)
    if retained is not None:
        active.scatter_(1, retained[:, None], False)
    return dict(
        parents=parents,
        targets=selected,
        target_mask=valid,
        matched=matched,
        active=active,
        target_indices=idx,
    )


def residual_target(target, parent):
    d = target.float() - parent.float()
    return torch.cat((d[..., :2], wrap(d[..., 2:])), dim=-1)


def paired_wta(result, heading_weight=0.25, unmatched_weight=0.05):
    distance = trajectory_distance(
        result["proposals"], result["targets"], heading_weight
    )
    winner = distance.detach().masked_fill(~result["target_mask"], torch.inf).argmin(-1)
    regression = distance.gather(-1, winner[..., None]).squeeze(-1)
    velocity = (
        (result["pred_velocity"].float() - result["target_velocity"].float())
        .square()
        .mean((-1, -2))
    )
    velocity = velocity.gather(-1, winner[..., None]).squeeze(-1)
    losses = {}
    for name, group in [
        ("matched", result["matched"]),
        ("unmatched", ~result["matched"]),
    ]:
        group = group & result["active"]
        count = group.sum(-1)
        losses[name + "_velocity"] = (
            (velocity * group).sum(-1) / count.clamp_min(1)
        ).mean()
        losses[name + "_trajectory"] = (
            (regression * group).sum(-1) / count.clamp_min(1)
        ).mean()
        losses[name + "_parents"] = count.float().mean()
    losses["velocity"] = (
        losses["matched_velocity"] + unmatched_weight * losses["unmatched_velocity"]
    )
    losses["trajectory"] = (
        losses["matched_trajectory"] + unmatched_weight * losses["unmatched_trajectory"]
    )
    return losses
