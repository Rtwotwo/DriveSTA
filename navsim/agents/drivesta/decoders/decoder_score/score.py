# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAVSIM (https://github.com/valeoai/DrivoR)
# Copyright (c) Valeoai-Model. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn

from navsim.agents.drivesta.layers.mlp import MLP
from navsim.agents.drivesta.layers.transformer import TransformerBlock


class ScoreTransformer(nn.Module):
    def __init__(self, num_layers, d_model, proj_drop, drop_path, config):
        super().__init__()

        _layers = []
        for _ in range(num_layers):
            _layers.append(
                TransformerBlock(
                    dim=d_model,
                    num_heads=(
                        config.refiner_num_heads
                        if hasattr(config, "refiner_num_heads")
                        else 1
                    ),
                    init_values=(
                        config.refiner_ls_values
                        if hasattr(config, "refiner_ls_values")
                        else 0.0
                    ),
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                )
            )
        self.layers = nn.ModuleList(_layers)
        self.return_intermediate = False

    def forward(self, x, x_cross):
        intermediate = []
        for _, layer in enumerate(self.layers):
            x = layer(x, x_cross)
            if self.return_intermediate:
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return x


class ModeCrossAttentionBlock(nn.Module):
    """Cross-attention block over proposal/mode tokens."""

    def __init__(self, d_model: int, d_ffn: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, proposal_feature: torch.Tensor) -> torch.Tensor:
        x = proposal_feature
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h2 = self.norm2(x)
        x = x + self.ffn(h2)
        return x


class ScoreDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.proposal_num = config.proposal_num
        self.poses_num = config.num_poses
        self.tf_d_model = config.tf_d_model
        self.one_token_per_traj = config.one_token_per_traj
        self.score_num = 6
        self.noc_weight = config.noc
        self.dac_weight = config.dac
        self.ddc_weight = config.ddc
        self.ttc_weight = config.ttc
        self.ep_weight = config.ep
        self.comfort_weight = config.comfort

        self.scorer_detach_proposals = bool(
            getattr(config, "scorer_detach_proposals", True)
        )
        self.scorer_use_serial_vlm_context = bool(
            getattr(config, "scorer_use_serial_vlm_context", False)
        )
        self.score_transformer = ScoreTransformer(
            num_layers=config.scorer_ref_num,
            d_model=config.tf_d_model,
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(self.poses_num * 3, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        self.pred_score = nn.ModuleDict(
            {
                "no_at_fault_collisions": self._score_head(config),
                "drivable_area_compliance": self._score_head(config),
                "time_to_collision_within_bound": self._score_head(config),
                "ego_progress": self._score_head(config),
                "driving_direction_compliance": self._score_head(config),
                "comfort": self._score_head(config),
            }
        )

        if bool(getattr(config, "use_diffusion_decoder", False)):
            self.jepa_norm = nn.LayerNorm(config.tf_d_model)
            self.jepa_attention = nn.MultiheadAttention(
                config.tf_d_model, 8, dropout=0.0, batch_first=True
            )
            self.jepa_gate = nn.Parameter(torch.zeros(()))
        self.double_score = config.double_score
        if self.double_score:
            self.pred_score2 = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        self.agent_pred = config.agent_pred
        if self.agent_pred:
            agent_dim = 2 * 40 * 9
            self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, agent_dim)

        self.area_pred = config.area_pred
        if self.area_pred:
            self.pred_area = self._build_area_head(config)

        self.bev_map = config.bev_map
        self.bev_agent = config.bev_agent

        if config.bev_agent:
            raise NotImplementedError

        if config.bev_map:
            raise NotImplementedError

        self._use_mode_cross_attn = bool(
            getattr(config, "scorer_use_mode_cross_attention", False)
        )
        if self._use_mode_cross_attn:
            num_heads = getattr(config, "scorer_mode_cross_attn_heads", 8)
            score_cross_dropout = float(getattr(config, "score_cross_dropout", 0.1))
            self.mode_cross_attn = ModeCrossAttentionBlock(
                d_model=self.tf_d_model,
                d_ffn=config.tf_d_ffn,
                num_heads=num_heads,
                dropout=score_cross_dropout,
            )

    @staticmethod
    def _score_head(config):
        return nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, 1),
        )

    def _build_area_head(self, config):
        if config.one_token_per_traj:
            area_dim = config.num_poses * 5 * 2
        else:
            area_dim = 5 * 2
        return MLP(config.tf_d_model, config.tf_d_ffn, area_dim)

    def _encode_score_tokens(
        self,
        proposals: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
        video_memory: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, proposal_num, _, _ = proposals.shape
        scorer_input_proposals = (
            proposals.detach() if self.scorer_detach_proposals else proposals
        )
        embedded_traj = self.pos_embed(
            scorer_input_proposals.reshape(batch_size, proposal_num, -1)
        )
        score_tokens = self.score_transformer(embedded_traj, scene_features)
        score_tokens = score_tokens + ego_token
        if video_memory is not None:  # local legacy argument name; this is JEPA memory
            q = self.jepa_norm(score_tokens)
            extra = self.jepa_attention(
                q, video_memory, video_memory, need_weights=False
            )[0]
            score_tokens = score_tokens + self.jepa_gate.tanh() * extra
        return score_tokens

    def _decode_score_tokens(
        self,
        proposals: torch.Tensor,
        score_tokens: torch.Tensor,
    ) -> dict:
        batch_size = proposals.shape[0]
        proposal_num = proposals.shape[1]
        pose_num = proposals.shape[2]

        proposal_feature = score_tokens
        if self._use_mode_cross_attn:
            proposal_feature = self.mode_cross_attn(proposal_feature)

        pred_logit = {}
        for key, head in self.pred_score.items():
            pred_logit[key] = head(proposal_feature).squeeze(-1)

        pred_logit2 = None
        pred_agents_states = None
        pred_area_logit = None
        bev_semantic_map = None
        agent_states = None
        agent_labels = None

        if self.double_score:
            pred_logit2 = self.pred_score2(proposal_feature).reshape(
                batch_size,
                -1,
                self.score_num,
            )

        if self.training:
            if self.area_pred:
                pred_area_logit = self.pred_area(score_tokens)
                if self.one_token_per_traj:
                    pred_area_logit = pred_area_logit.reshape(
                        batch_size,
                        proposal_num,
                        pose_num,
                        -1,
                    )

            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(proposal_feature).reshape(
                    batch_size,
                    proposal_num,
                    pose_num,
                    -1,
                    2,
                    9,
                )

            if self.bev_map:
                bev_semantic_map = self.map_head(score_tokens)

            if self.bev_agent:
                agents = self._agent_head(None, score_tokens)
                agent_states = agents[:, :, :-1]
                agent_labels = agents[:, :, -1]

        return {
            "pred_logit": pred_logit,
            "pred_logit2": pred_logit2,
            "pred_agents_states": pred_agents_states,
            "pred_area_logit": pred_area_logit,
            "bev_semantic_map": bev_semantic_map,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
        }

    def _compute_pdm_score(self, pred_logit: dict) -> torch.Tensor:
        p = {name: value.float().sigmoid() for name, value in pred_logit.items()}
        denominator = self.ttc_weight + self.ep_weight + self.comfort_weight
        if denominator <= 0:
            raise ValueError("PDM weighted metric denominator must be positive")
        return (
            p["no_at_fault_collisions"].pow(self.noc_weight)
            * p["drivable_area_compliance"].pow(self.dac_weight)
            * p["driving_direction_compliance"].pow(self.ddc_weight)
            * (
                self.ttc_weight * p["time_to_collision_within_bound"]
                + self.ep_weight * p["ego_progress"]
                + self.comfort_weight * p["comfort"]
            )
            / denominator
        )

    def forward(
        self,
        proposals: torch.Tensor,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
        video_memory: Optional[torch.Tensor] = None,
    ) -> dict:
        score_tokens = self._encode_score_tokens(
            proposals,
            scene_features,
            ego_token,
            video_memory,
        )
        score_output = self._decode_score_tokens(proposals, score_tokens)
        score_output["pdm_score"] = self._compute_pdm_score(score_output["pred_logit"])
        return score_output
