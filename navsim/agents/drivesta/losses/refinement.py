"""Expert-anchored generation and separate officially labelled score views."""

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

COMPONENTS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def compose_pdm(p):
    return p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12


def trajectory_distance(pred, target, heading_weight=0.25):
    d = pred.float() - target.float()
    return (
        d[..., :2].abs().sum(-1)
        + heading_weight * torch.atan2(d[..., 2].sin(), d[..., 2].cos()).abs()
    ).mean(-1)


def expert_loss(proposals, target):
    # Preserve the original planner's best-of-64 L1, including raw heading L1.
    return (
        (proposals.float() - target[:, None].float())
        .abs()
        .sum(-1)
        .mean(-1)
        .amin(1)
        .mean()
    )


def auxiliary_loss(
    proposals, trajectories, mask, heading_weight=0.25, matching="injective"
):
    if matching not in ("injective", "nearest"):
        raise ValueError("Unknown auxiliary matching")
    losses = []
    for p, h, valid in zip(proposals, trajectories, mask.bool()):
        h = h[valid]
        if not len(h):
            losses.append(p.float().sum() * 0)
            continue
        if len(h) > len(p):
            raise ValueError("More auxiliary targets than predictions")
        cost = trajectory_distance(p[:, None], h[None], heading_weight)
        if matching == "nearest":
            losses.append(cost.amin(0).mean())
        else:
            # Only the discrete assignment is detached; matched distances retain gradients.
            rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
            rows = torch.as_tensor(rows, device=p.device)
            cols = torch.as_tensor(cols, device=p.device)
            losses.append(cost[rows, cols].mean())
    return torch.stack(losses).mean()


def frontier_rank_loss(
    prediction, truth, mask, top_k=8, gap_min=0.005, temperature=0.05
):
    if top_k < 1 or temperature <= 0 or gap_min < 0:
        raise ValueError("Invalid ranking configuration")
    with torch.no_grad():
        truth = truth.detach()
        valid = mask.bool()
        k = min(top_k, prediction.shape[1])
        top = torch.zeros_like(valid)
        top.scatter_(
            1,
            prediction.detach().masked_fill(~valid, -torch.inf).topk(k, dim=1).indices,
            True,
        )
        top.scatter_(
            1, truth.masked_fill(~valid, -torch.inf).topk(k, dim=1).indices, True
        )
        top &= valid
        gap = truth[:, :, None] - truth[:, None, :]
        pair = (
            valid[:, :, None]
            & valid[:, None, :]
            & (top[:, :, None] | top[:, None, :])
            & (gap > gap_min)
        )
        weight = gap.clamp_min(0) * pair
    terms = F.softplus(-(prediction[:, :, None] - prediction[:, None, :]) / temperature)
    loss = ((terms * weight).sum((1, 2)) / weight.sum((1, 2)).clamp_min(1e-8)).mean()
    return loss, pair.sum((1, 2)).float().mean()


def score_loss(
    logits_dict,
    labels,
    mask,
    total_weight=0.5,
    rank_weight=0.0,
    rank_top_k=8,
    rank_gap_min=0.005,
    rank_temperature=0.05,
):
    logits = torch.stack([logits_dict[k] for k in COMPONENTS], -1).float()
    labels = labels.float()
    mask = mask.bool()
    if logits.shape != labels[..., :6].shape or mask.shape != labels.shape[:2]:
        raise ValueError("Scorer labels/context mask mismatch")
    if not torch.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
        raise ValueError("Invalid official score labels")
    count = mask.sum(-1).clamp_min(1)
    component = F.binary_cross_entropy_with_logits(
        logits, labels[..., :6], reduction="none"
    ).sum(-1)
    component = ((component * mask).sum(-1) / count).mean()
    prediction = compose_pdm(logits.sigmoid())
    mse = (((prediction - labels[..., 6]).square() * mask).sum(-1) / count).mean()
    if rank_weight:
        rank, pairs = frontier_rank_loss(
            prediction, labels[..., 6], mask, rank_top_k, rank_gap_min, rank_temperature
        )
    else:
        rank = pairs = prediction.new_zeros(())
    with torch.no_grad():
        chosen = prediction.masked_fill(~mask, -torch.inf).argmax(-1)
        best = labels[..., 6].masked_fill(~mask, -torch.inf).amax(-1)
        selected = labels[..., 6].gather(1, chosen[:, None]).squeeze(1)
        regret = torch.where(mask.any(-1), best - selected, 0.0).mean()
        mae = (((prediction - labels[..., 6]).abs() * mask).sum(-1) / count).mean()
        selected = torch.where(mask.any(-1), selected, 0.0).mean()
        best = torch.where(mask.any(-1), best, 0.0).mean()
        per_component = (
            ((logits.sigmoid() - labels[..., :6]).abs() * mask[..., None]).sum(1)
            / count[:, None]
        ).mean(0)
    return component + total_weight * mse + rank_weight * rank, dict(
        component=component,
        total_mse=mse,
        rank=rank,
        mae=mae,
        regret=regret,
        rank_pairs=pairs,
        selected=selected,
        oracle=best,
        component_mae=per_component,
    )


class DriveSTALoss(nn.Module):
    def __init__(
        self,
        phase="joint",
        heading_weight=0.25,
        unmatched_weight=0.05,
        diffusion_loss_weight=10.0,
        diffusion_trajectory_loss_weight=20.0,
        replay_weight=0.25,
        total_score_weight=0.5,
        rank_weight=0.05,
        rank_warmup_epochs=2.0,
        rank_top_k=8,
        rank_gap_min=0.005,
        rank_temperature=0.05,
        auxiliary_weight=0.25,
        auxiliary_warmup_epochs=2.0,
        ar_auxiliary_samples=8,
        future_weight=1.0,
        mask_unavailable_long_gt=False,
    ):
        super().__init__()
        if phase != "joint" or ar_auxiliary_samples != 8:
            raise ValueError("Invalid Refinement phase or unsupported AR target count")
        if not future_weight > 0:
            raise ValueError("V4 requires positive expert-future supervision")
        for key, value in locals().copy().items():
            if key not in ("self", "__class__"):
                setattr(self, key, value)
        if rank_weight < 0:
            raise ValueError("rank_weight must be nonnegative")
        self.epoch_progress = 0.0

    def protocol(self):
        protocol = {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_")
            and k not in ("epoch_progress", "training", "mask_unavailable_long_gt")
            and not isinstance(v, dict)
        }
        if self.mask_unavailable_long_gt:
            protocol["long_gt"] = "real_only_batch_mean_mask"
        return protocol

    def forward(self, targets, pred):
        if self.mask_unavailable_long_gt:
            valid = targets["trajectory_long_valid"]
            if (
                valid.dtype != torch.bool
                or valid.shape != targets["trajectory"].shape[:1]
            ):
                raise ValueError("Expected one long-GT validity flag per scene")
            targets = dict(targets)
            targets["trajectory_long"] = torch.where(
                valid[:, None, None],
                targets["trajectory_long"].float(),
                pred["ar_proposals"][:, 0].detach().float(),
            )
        if self.phase in ("dit", "joint"):
            from navsim.agents.drivesta.losses.residual_wta import paired_wta

            w = paired_wta(
                pred["diffusion_training"], self.heading_weight, self.unmatched_weight
            )
            loss = (
                self.diffusion_loss_weight * w["velocity"]
                + self.diffusion_trajectory_loss_weight * w["trajectory"]
            )
            dit_out = dict(
                loss=loss,
                **{"dit_" + k: v for k, v in w.items()},
                dense_targets=targets["auxiliary_mask"].sum(-1).float().mean(),
            )
            if self.phase == "dit":
                return dit_out
        scale = min(1.0, self.epoch_progress / max(self.rank_warmup_epochs, 1e-8))
        native, n = score_loss(
            pred["pred_logit"],
            targets["native_labels"],
            torch.ones_like(targets["native_labels"][..., 0], dtype=torch.bool),
            self.total_score_weight,
            self.rank_weight * scale,
            self.rank_top_k,
            self.rank_gap_min,
            self.rank_temperature,
        )
        replay, r = score_loss(
            pred["replay_pred_logit"],
            targets["replay_labels"],
            targets["replay_mask"],
            self.total_score_weight,
        )
        out = dict(
            loss=native + self.replay_weight * replay,
            native_score_loss=native,
            replay_loss=replay,
            native_pdm_mae=n["mae"],
            native_selection_regret=n["regret"],
            native_selected_pdms=n["selected"] * 100,
            native_oracle_pdms=n["oracle"] * 100,
            native_rank=n["rank"],
            native_total_mse=n["total_mse"],
        )
        for name, value in zip(COMPONENTS, n["component_mae"]):
            out["native_" + name + "_mae"] = value
        if self.phase == "joint":
            ar = pred["ar_proposals"]
            anchor = expert_loss(ar, targets["trajectory"]) + expert_loss(
                ar, targets["trajectory_long"]
            )
            # The dataset supplies a separate epoch-rotating 8-of-32 AR view.
            aux = auxiliary_loss(
                ar,
                targets["ar_auxiliary_trajectories"],
                targets["ar_auxiliary_mask"],
                self.heading_weight,
            )
            scale = min(
                1.0, self.epoch_progress / max(self.auxiliary_warmup_epochs, 1e-8)
            )
            out["loss"] = (
                out["loss"]
                + dit_out.pop("loss")
                + anchor
                + self.auxiliary_weight * scale * aux
            )
            out.update(
                dit_out,
                expert_loss=anchor,
                auxiliary_loss=aux,
                auxiliary_scale=anchor.detach() * 0 + scale,
                ar_auxiliary_count=targets["ar_auxiliary_mask"].sum(-1).float().mean(),
            )
        future = pred["jepa_future"]
        out["loss"] = out["loss"] + self.future_weight * future["loss"]
        out.update({"jepa_future_" + key: value for key, value in future.items()})
        if self.mask_unavailable_long_gt:
            out["long_gt_valid_fraction"] = valid.float().mean()
        return out
