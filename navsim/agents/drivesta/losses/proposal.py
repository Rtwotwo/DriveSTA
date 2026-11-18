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
    rank, pairs = frontier_rank_loss(
        prediction, labels[..., 6], mask, rank_top_k, rank_gap_min, rank_temperature
    )
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


class DriveSTAProposalLoss(nn.Module):
    def __init__(
        self,
        auxiliary_weight=0.25,
        auxiliary_warmup_epochs=2.0,
        heading_weight=0.25,
        replay_weight=0.25,
        total_score_weight=0.5,
        rank_weight=0.05,
        diffusion_loss_weight=10.0,
        diffusion_trajectory_loss_weight=20.0,
        matching="injective",
        rank_warmup_epochs=2.0,
        rank_top_k=8,
        rank_gap_min=0.005,
        rank_temperature=0.05,
        mask_unavailable_long_gt=False,
    ):
        super().__init__()
        self.auxiliary_weight = auxiliary_weight
        self.auxiliary_warmup_epochs = auxiliary_warmup_epochs
        self.heading_weight = heading_weight
        self.replay_weight = replay_weight
        self.total_score_weight = total_score_weight
        self.rank_weight = rank_weight
        self.diffusion_loss_weight = diffusion_loss_weight
        self.diffusion_trajectory_loss_weight = diffusion_trajectory_loss_weight
        self.matching = matching
        self.rank_warmup_epochs = rank_warmup_epochs
        self.rank_top_k = rank_top_k
        self.rank_gap_min = rank_gap_min
        self.rank_temperature = rank_temperature
        self.mask_unavailable_long_gt = bool(mask_unavailable_long_gt)
        if rank_weight < 0:
            raise ValueError("rank_weight must be nonnegative")
        self.epoch_progress = 0.0

    def protocol(self):
        keys = (
            "auxiliary_weight",
            "auxiliary_warmup_epochs",
            "heading_weight",
            "replay_weight",
            "total_score_weight",
            "rank_weight",
            "diffusion_loss_weight",
            "diffusion_trajectory_loss_weight",
            "matching",
            "rank_warmup_epochs",
            "rank_top_k",
            "rank_gap_min",
            "rank_temperature",
        )
        protocol = dict(
            version=2,
            generator="expert_and_rotating_screened_matching",
            scorer="native64_frontier_plus_rotating64",
            **{k: getattr(self, k) for k in keys},
        )
        if self.mask_unavailable_long_gt:
            protocol["long_gt"] = "real_only_batch_mean_mask"
        return protocol

    def forward(self, targets, pred):
        ar = pred["proposal_list"][-1]
        anchor = expert_loss(ar, targets["trajectory"]) + expert_loss(
            ar, targets["trajectory_long"]
        )
        aux = auxiliary_loss(
            pred["proposals"],
            targets["auxiliary_trajectories"],
            targets["auxiliary_mask"],
            self.heading_weight,
            self.matching,
        )
        scale = min(1.0, self.epoch_progress / max(self.auxiliary_warmup_epochs, 1e-8))
        rank_scale = min(1.0, self.epoch_progress / max(self.rank_warmup_epochs, 1e-8))
        zero = anchor * 0
        denoise = regression = zero
        for item in pred.get("diffusion_aux_list", []):
            b = torch.arange(len(ar), device=ar.device)
            i = item["winner_idx"]
            denoise = denoise + F.mse_loss(
                item["pred_delta_normed"][b, i].float(),
                item["target_delta_normed"][b, i].float(),
            )
            regression = expert_loss(pred["proposals"], targets["trajectory"])
        native, n = score_loss(
            pred["pred_logit"],
            targets["native_labels"],
            torch.ones_like(targets["native_labels"][..., 0], dtype=torch.bool),
            self.total_score_weight,
            self.rank_weight * rank_scale,
            self.rank_top_k,
            self.rank_gap_min,
            self.rank_temperature,
        )
        replay, r = score_loss(
            pred["replay_pred_logit"],
            targets["replay_labels"],
            targets["replay_mask"],
            self.total_score_weight,
            0.0,
        )
        loss = (
            anchor
            + self.auxiliary_weight * scale * aux
            + self.diffusion_loss_weight * denoise
            + self.diffusion_trajectory_loss_weight * regression
            + native
            + self.replay_weight * replay
        )
        result = dict(
            loss=loss,
            expert_loss=anchor,
            auxiliary_loss=aux,
            auxiliary_scale=zero + scale,
            diffusion_loss=denoise,
            diffusion_trajectory_loss=regression,
            native_score_loss=native,
            replay_score_loss=replay,
            native_pdm_mae=n["mae"],
            native_selection_regret=n["regret"],
            native_total_mse=n["total_mse"],
            replay_pdm_mae=r["mae"],
            replay_total_mse=r["total_mse"],
            auxiliary_count=targets["auxiliary_mask"].float().sum(-1).mean(),
            native_high_fraction=(targets["native_labels"][..., 6] >= 0.95)
            .float()
            .mean(),
            native_rank_loss=n["rank"],
            native_rank_pairs=n["rank_pairs"],
            rank_scale=zero + rank_scale,
            native_selected_pdms=100 * n["selected"],
            native_oracle_pdms=100 * n["oracle"],
        )
        if "auxiliary_bank_size" in targets:
            result["auxiliary_bank_size"] = (
                targets["auxiliary_bank_size"].float().mean()
            )
        for i, name in enumerate(COMPONENTS):
            result["native_" + name + "_mae"] = n["component_mae"][i]
            result["replay_" + name + "_mae"] = r["component_mae"][i]
        result["weighted_replay_loss"] = self.replay_weight * replay.detach()
        result["weighted_dense_loss"] = self.auxiliary_weight * scale * aux.detach()
        result["weighted_rank_loss"] = (
            self.rank_weight * rank_scale * n["rank"].detach()
        )
        if self.mask_unavailable_long_gt:
            valid = targets["trajectory_long_valid"].bool()
            if not valid.all():
                ar = pred["proposal_list"][-1].float()
                per_scene = (
                    (ar - targets["trajectory_long"][:, None].float())
                    .abs()
                    .sum(-1)
                    .mean(-1)
                    .amin(1)
                )
                unavailable = (per_scene * (~valid)).mean()
                result["loss"] = result["loss"] - unavailable
                result["expert_loss"] = result["expert_loss"] - unavailable
        return result
