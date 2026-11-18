"""Causal observed-video conditioning; no future-image or VLM input."""

import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
from navsim.agents.drivesta.encoders.jepa_vendor.models.vision_transformer import (
    vit_large,
)


def preprocess_front(images):
    import cv2

    if len(images) not in (2, 8) or any(x is None or x.ndim != 3 for x in images):
        raise ValueError(
            "JEPA requires 2 observed frames or 8 real expert video frames"
        )
    frames = [
        cv2.resize(x[28:-28], (512, 256), interpolation=cv2.INTER_LINEAR)
        for x in images
    ]
    x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0
    return (
        x - x.new_tensor([0.485, 0.456, 0.406])[:, None, None, None]
    ) / x.new_tensor([0.229, 0.224, 0.225])[:, None, None, None]


def official_masks(seeds, device):
    """Published two mask families; original sampler, deterministic CPU RNG."""
    from navsim.agents.drivesta.encoders.jepa_vendor.masks.multiseq_multiblock3d import (
        _MaskGenerator,
    )

    seed = int(seeds.detach().cpu().sum()) % (2**31 - 3)
    result = []
    for group, (blocks, scale) in enumerate(((8, 0.15), (2, 0.7))):
        generator = _MaskGenerator(
            crop_size=(256, 512),
            num_frames=8,
            spatial_patch_size=(16, 16),
            temporal_patch_size=2,
            spatial_pred_mask_scale=(scale, scale),
            temporal_pred_mask_scale=(1.0, 1.0),
            aspect_ratio=(0.75, 1.5),
            npred=blocks,
        )
        generator._itr_counter.value = seed + group - 1
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + group)
            context, target = generator(len(seeds))
        result.append((context.to(device), target.to(device)))
    return result


class TrainableJEPA(nn.Module):
    """Expert-video masked pretraining, strictly separated from planner inputs."""

    def __init__(self, checkpoint, cfg, encoder=None, predictor=None):
        super().__init__()
        from navsim.agents.drivesta.encoders.jepa_vendor.models.predictor import (
            vit_predictor,
        )

        self.encoder = (
            encoder
            if encoder is not None
            else vit_large(
                img_size=(256, 512),
                num_frames=8,
                tubelet_size=2,
                use_rope=True,
                uniform_power=True,
                handle_nonsquare_inputs=True,
                use_activation_checkpointing=bool(cfg.jepa_activation_checkpointing),
            )
        )
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.target_encoder.use_activation_checkpointing = False
        self.predictor = (
            predictor
            if predictor is not None
            else vit_predictor(
                img_size=(256, 512),
                num_frames=8,
                patch_size=16,
                tubelet_size=2,
                embed_dim=1024,
                predictor_embed_dim=int(cfg.jepa_predictor_dim),
                depth=int(cfg.jepa_predictor_layers),
                num_heads=int(cfg.jepa_predictor_heads),
                use_mask_tokens=True,
                num_mask_tokens=int(cfg.jepa_mask_tokens),
                use_rope=True,
                uniform_power=True,
                use_activation_checkpointing=bool(cfg.jepa_activation_checkpointing),
            )
        )
        if encoder is None and checkpoint is not None:
            state = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            for name in ("encoder", "target_encoder", "predictor"):
                clean = {
                    k.removeprefix("module.").removeprefix("backbone."): v
                    for k, v in state[name].items()
                }
                getattr(self, name).load_state_dict(clean, strict=True)
        self.momentum = float(cfg.jepa_ema_momentum)
        if not 0 <= self.momentum < 1:
            raise ValueError("Invalid EMA momentum")
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self):
        if not any(p.grad is not None for p in self.encoder.parameters()):
            return
        online = dict(self.encoder.named_parameters())
        for name, p in self.target_encoder.named_parameters():
            p.lerp_(online[name], 1 - self.momentum)
        online_buffers = dict(self.encoder.named_buffers())
        for name, b in self.target_encoder.named_buffers():
            b.copy_(online_buffers[name])
        self.ema_updates.add_(1)

    @torch.no_grad()
    def forward(self, clip):
        # Planner forward: no future pixels, no graph or planning update to JEPA.
        if clip.shape[1:] != (3, 2, 256, 512) or not torch.isfinite(clip).all():
            raise ValueError("JEPA input must be finite [B,3,2,256,512] observed RGB")
        tokens = self.encoder(clip)
        if tokens.shape[1:] != (512, 1024):
            raise ValueError(f"Unexpected JEPA memory {tokens.shape}")
        return tokens

    def future_objective(self, targets):
        from navsim.agents.drivesta.encoders.jepa_vendor.masks.utils import apply_masks

        valid = targets["jepa_future_valid"].bool()
        count = valid.sum().detach().float()
        global_count = count.clone()
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            torch.distributed.all_reduce(global_count)
        # DDP gradient averaging must produce a GLOBAL valid-example mean.
        scale = count * world / global_count.clamp_min(1)
        zero = count.new_zeros(())
        names = (
            "cosine",
            "target_std",
            "prediction_std",
            "context_tokens",
            "target_tokens",
        )
        if not valid.any():
            return dict(
                loss=zero,
                valid_fraction=valid.float().mean(),
                **{k: zero for k in names},
            )
        clip = targets["jepa_video_clip"][valid].float()
        offsets = targets["jepa_video_offsets"][valid].float()
        if clip.shape[1:] != (3, 8, 256, 512) or not torch.isfinite(clip).all():
            raise ValueError("Need finite real expert video [B,3,8,256,512]")
        if (
            offsets.shape[1:] != (8,)
            or not torch.isfinite(offsets).all()
            or not (offsets[:, -1] > 0).all()
            or not ((offsets[:, 1:] - offsets[:, :-1] - 0.5).abs() <= 0.100001).all()
        ):
            raise ValueError("Misaligned 2Hz expert video timestamps")
        masks = official_masks(targets["jepa_mask_seed"][valid], clip.device)
        # BF16 follows official train.py; normalization and L1 use FP32.
        with torch.no_grad():
            target = F.layer_norm(self.target_encoder(clip).float(), (1024,))
        losses, metrics = [], {k: [] for k in names}
        for group, (context_indices, target_indices) in enumerate(masks):
            context = self.encoder(clip, masks=context_indices)
            prediction = self.predictor(
                context, context_indices, target_indices, mask_index=group
            ).float()
            expected_target = apply_masks(target, [target_indices])
            losses.append(F.l1_loss(prediction, expected_target))
            with torch.no_grad():
                metrics["cosine"].append(
                    F.cosine_similarity(prediction, expected_target, dim=-1).mean()
                )
                metrics["target_std"].append(
                    expected_target.flatten(0, 1).std(0, unbiased=False).mean()
                )
                metrics["prediction_std"].append(
                    prediction.flatten(0, 1).std(0, unbiased=False).mean()
                )
                metrics["context_tokens"].append(
                    count.new_tensor(context_indices.shape[1])
                )
                metrics["target_tokens"].append(
                    count.new_tensor(target_indices.shape[1])
                )
        return dict(
            loss=torch.stack(losses).mean() * scale,
            valid_fraction=valid.float().mean(),
            **{k: torch.stack(v).mean() * scale for k, v in metrics.items()},
        )


class VideoCondition(nn.Module):
    def __init__(self, dim=256, pooled=False):
        super().__init__()
        self.pooled = pooled
        self.project = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, dim))
        self.space = nn.Parameter(torch.randn(1, 16, 32, dim) * 0.02)
        self.motion = nn.Sequential(nn.Linear(44, dim), nn.GELU(), nn.Linear(dim, dim))
        self.motion_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, tokens, ego_history):
        if ego_history.shape[1:] != (4, 11):
            raise ValueError(
                "Need four historical ego states including motion/navigation"
            )
        x = (
            self.project(tokens.float()).reshape(-1, 16, 32, self.space.shape[-1])
            + self.space
        )
        if self.pooled:
            x = F.avg_pool2d(x.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        x = x.flatten(1, 2)
        motion = self.motion(ego_history.float().flatten(1))[:, None] + self.motion_type
        return torch.cat((x, motion), dim=1)
