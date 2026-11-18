from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from navsim.agents.drivesta.utils import pylogger

from navsim.agents.drivesta.decoders.decoder_score.score import ScoreDecoder
from navsim.agents.drivesta.decoders.trajectory import DriveSTATrajectoryDecoder
from navsim.agents.drivesta.encoders.image.dinov2_lora import ImgEncoder


log = pylogger.get_pylogger(__name__)


def score_replay(
    scorer, proposals, mask, scene_features, ego_token, vlm_hidden_state=None
):
    if mask.all():
        return scorer(proposals, scene_features, ego_token, vlm_hidden_state)[
            "pred_logit"
        ]
    rows = []
    for b in range(len(proposals)):
        count = int(mask[b].sum())
        if count < 1:
            raise ValueError("Scorer replay has no valid candidate")
        if not mask[b, :count].all() or mask[b, count:].any():
            raise ValueError("Replay mask must be prefix-valid")
        result = scorer(
            proposals[b : b + 1, :count],
            scene_features[b : b + 1],
            ego_token[b : b + 1],
            None if vlm_hidden_state is None else vlm_hidden_state[b : b + 1],
        )["pred_logit"]
        rows.append(
            {
                k: torch.nn.functional.pad(v, (0, proposals.shape[1] - count))
                for k, v in result.items()
            }
        )
    return {k: torch.cat([r[k] for r in rows]) for k in rows[0]}


CAMERA_CONFIG_KEYS: Tuple[str, ...] = (
    "cam_f0",
    "cam_l0",
    "cam_l1",
    "cam_l2",
    "cam_r0",
    "cam_r1",
    "cam_r2",
    "cam_b0",
)


class DriveSTAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model
        self.b2d = bool(getattr(config, "b2d", False))

        self.num_cams = self._count_enabled_sensors(CAMERA_CONFIG_KEYS)
        self.num_lidar = 1 if len(self._config["lidar_pc"]) > 0 else 0

        self._build_scene_backbones(config)
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11 * 4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        self.trajectory_decoder = DriveSTATrajectoryDecoder(config)
        self._use_diffusion_decoder = bool(
            getattr(config, "use_diffusion_decoder", False)
        )
        self._train_scorer = bool(getattr(config, "train_scorer", True))
        self.score_decoder = ScoreDecoder(config)

        if self._use_diffusion_decoder:
            import copy
            from navsim.agents.drivesta.encoders.jepa import TrainableJEPA, VideoCondition

            self.reference_scorer = copy.deepcopy(self.score_decoder)
            self.jepa = TrainableJEPA(
                config.jepa_checkpoint if config.jepa_load_pretrained else None, config
            )
            self.dit_condition = VideoCondition(config.tf_d_model, pooled=True)
            self.score_condition = VideoCondition(config.tf_d_model, pooled=False)
            # Capture the original LoRA/base-backbone mask before phase freezing.
            self._joint_trainable = {
                name
                for name, p in self.named_parameters()
                if p.requires_grad
                and not name.startswith(("jepa.target_encoder.", "reference_scorer."))
            }
            self.phase = str(config.training_phase)
            if self.phase != "joint":
                raise ValueError("Unknown Refinement training phase")
            self.configure_phase()

        else:
            self._configure_trainable_parameters()

    def _count_enabled_sensors(self, keys: Tuple[str, ...]) -> int:
        return sum(1 for key in keys if len(self._config[key]) > 0)

    def _build_scene_backbones(self, config) -> None:
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(
                torch.randn(
                    1,
                    self.num_cams,
                    self._config.num_scene_tokens,
                    self.image_backbone.num_features,
                )
                * 1e-6,
                requires_grad=True,
            )

        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            lidar_num_features = getattr(
                self,
                "image_backbone",
                self.lidar_backbone,
            ).num_features
            self.lidar_scene_embeds = nn.Parameter(
                torch.randn(
                    1,
                    self.num_lidar,
                    self._config.num_scene_tokens,
                    lidar_num_features,
                )
                * 1e-6,
                requires_grad=True,
            )

    def _configure_trainable_parameters(self) -> None:
        train_diffusion_decoder_only = bool(
            getattr(self._config, "train_diffusion_decoder_only", False)
        )
        if not train_diffusion_decoder_only:
            self._set_score_decoder_trainable(self._train_scorer)
            return

        if not self._use_diffusion_decoder:
            log.warning(
                "train_diffusion_decoder_only=True but diffusion decoder is disabled; skip freeze."
            )
            self._set_score_decoder_trainable(self._train_scorer)
            return

        for p in self.parameters():
            p.requires_grad = False

        diffusion_decoder = self.trajectory_decoder.diffusion_decoder
        if diffusion_decoder is not None:
            for p in diffusion_decoder.parameters():
                p.requires_grad = True

        if self._train_scorer:
            self._set_score_decoder_trainable(True)

    def _set_score_decoder_trainable(self, trainable: bool) -> None:
        for p in self.score_decoder.parameters():
            p.requires_grad = trainable

    def scene_encoder(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ego_status_last = features["ego_status"][:, -1]
        if self._config.full_history_status:
            ego_status = features["ego_status"].flatten(-2)
        else:
            ego_status = ego_status_last
        log.debug(f"Ego Status - {ego_status.shape}")

        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")

        batch_size = ego_status.shape[0]
        scene_features: List[torch.Tensor] = []

        if self.num_cams > 0:
            if "image" in features:
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError("Expected image or camera_feature in features.")

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)
            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        encoded_scene = torch.cat(scene_features, dim=1)

        return {
            "ego_status_last": ego_status_last,
            "ego_status": ego_status,
            "ego_token": ego_token,
            "scene_features": encoded_scene,
        }

    def configure_phase(self):
        self.requires_grad_(False)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in self._joint_trainable)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not self._use_diffusion_decoder:
            return self
        self.reference_scorer.eval()
        self.jepa.train(mode)  # its EMA target stays in eval mode
        return self

    def rollout(self, parents, retained, encoded, memory, tokens):
        if tokens is None:
            raise ValueError("Deterministic Refinement rollout requires scenario tokens")
        dit = self.trajectory_decoder.diffusion_decoder
        was_training = dit.training
        dit.eval()
        try:
            with torch.no_grad(), torch.autocast(
                device_type=parents.device.type, enabled=False
            ):
                proposals = dit(
                    dict(
                        ar_proposals=parents.detach().float(),
                        scene_features=memory.detach().float(),
                        ego_status_feature=encoded["ego_status_last"][..., 3:].float(),
                        scenario_token=tokens,
                    )
                )["proposals"]
                if retained is not None:
                    proposals = proposals.clone()
                    b = torch.arange(len(parents), device=parents.device)
                    proposals[b, retained] = parents[b, retained]
        finally:
            dit.train(was_training)
        return proposals

    def forward(self, features, targets=None):
        if self._use_diffusion_decoder:
            return self.forward_joint(features, targets)
        return self.forward_proposals(features, targets)

    def forward_joint(self, features, targets=None):
        from navsim.agents.drivesta.losses.residual_wta import make_pairs

        encoded = self.scene_encoder(features)
        if encoded["scene_features"].shape[1:] != (64, 256):
            raise ValueError("V2 requires four cameras x 16 scene tokens")
        original = self.trajectory_decoder(
            encoded["ego_token"], encoded["scene_features"], encoded["ego_status_last"]
        )
        parents = original["proposals"]
        with torch.autocast(device_type=parents.device.type, enabled=False):
            video = self.jepa(features["jepa_clip"].float())
        memory = torch.cat(
            (
                encoded["scene_features"],
                self.dit_condition(video.detach(), features["ego_status"]),
            ),
            1,
        )
        # Deployment-like candidates have no access to training targets. Sampling
        # has no gradient; DiT learns via the separate paired denoising objective.
        proposals = self.rollout(
            parents, None, encoded, memory, features.get("scenario_token")
        )
        score_memory = self.score_condition(video.detach(), features["ego_status"])
        scored = self.score_decoder(
            proposals, encoded["scene_features"], encoded["ego_token"], score_memory
        )
        output = dict(
            proposals=proposals,
            ar_proposals=parents,
            proposal_list=original["proposal_list"],
            **scored,
        )
        output["trajectory"] = proposals[
            torch.arange(len(parents), device=parents.device),
            scored["pdm_score"].argmax(-1),
        ]
        if self.training:
            if targets is None:
                raise ValueError("Joint training requires dense and expert supervision")
            output["jepa_future"] = self.jepa.future_objective(targets)
            # A moving AR prediction must not change the noise regression target
            # through autograd. AR has its own expert/dense generation losses.
            pairs = make_pairs(parents.detach().float(), targets, None, self._config)
            output["diffusion_training"] = (
                self.trajectory_decoder.diffusion_decoder.forward_pairs(
                    pairs, memory, encoded["ego_status_last"][..., 3:]
                )
            )
            output["replay_pred_logit"] = score_replay(
                self.score_decoder,
                targets["replay_trajectories"],
                targets["replay_mask"],
                encoded["scene_features"],
                encoded["ego_token"],
                score_memory,
            )
        return output

    def forward_proposals(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded_features = self.scene_encoder(features)
        output = self.trajectory_decoder(
            ego_token=encoded_features["ego_token"],
            scene_features=encoded_features["scene_features"],
            ego_status_last=encoded_features["ego_status_last"],
            scenario_token=features.get("scenario_token"),
            targets=targets,
        )

        if self.training:
            if targets is None:
                raise ValueError("Training targets are required")
            output.update(
                self.score_decoder(
                    output.get("score_proposals", output["proposals"]),
                    encoded_features["scene_features"],
                    encoded_features["ego_token"],
                    features.get("vlm_hidden_state"),
                )
            )
            output["replay_pred_logit"] = score_replay(
                self.score_decoder,
                targets["replay_trajectories"],
                targets["replay_mask"],
                encoded_features["scene_features"],
                encoded_features["ego_token"],
                features.get("vlm_hidden_state"),
            )
            return output

        score_output = self.score_decoder(
            proposals=output["proposals"],
            scene_features=encoded_features["scene_features"],
            ego_token=encoded_features["ego_token"],
        )
        output.update(score_output)

        pdm_score = score_output["pdm_score"]
        selected_idx = torch.argmax(pdm_score, dim=1)
        output["trajectory"] = output["proposals"][
            torch.arange(pdm_score.shape[0], device=pdm_score.device),
            selected_idx,
        ]

        return output

