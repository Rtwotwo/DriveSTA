"""AR proposals and diffusion refinement with dense trajectory supervision."""

from typing import Optional
import torch
from torch import nn

from navsim.agents.drivesta.layers.transformer import TransformerLayer
from navsim.agents.drivesta.decoders.decoder_traj.ar import (
    AutoregressiveTrajectoryDecoder,
)
from navsim.agents.drivesta.decoders.decoder_traj.dit import DiffusionTrajectoryDecoder


class DriveSTATrajectoryDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._poses_num = config.num_poses
        self._proposal_num = config.proposal_num
        self._use_diffusion_decoder = bool(
            getattr(config, "use_diffusion_decoder", False)
        )
        self._eval_trajectory_source = str(
            getattr(config, "eval_trajectory_source", "diffusion")
        ).lower()
        if self._eval_trajectory_source not in {"ar", "diffusion"}:
            raise ValueError(
                f"Unsupported eval_trajectory_source: {self._eval_trajectory_source}"
            )

        self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)

        self.trajectory_transformer = TransformerLayer(
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
        )

        self.ar_decoders = nn.ModuleList(
            [AutoregressiveTrajectoryDecoder(config) for _ in range(config.ref_num + 1)]
        )
        self.diffusion_decoder: Optional[DiffusionTrajectoryDecoder]
        if self._use_diffusion_decoder:
            self.diffusion_decoder = DiffusionTrajectoryDecoder(config=config)
        else:
            self.diffusion_decoder = None

    def forward(
        self,
        ego_token,
        scene_features,
        ego_status_last,
        scenario_token=None,
        targets=None,
    ):
        tokens = ego_token + self.init_feature.weight[None]
        stages = [self.ar_decoders[0](tokens, scene_features, ego_status_last)]
        refined = self.trajectory_transformer(tokens, scene_features)
        for i in range(self._config.ref_num):
            stages.append(
                self.ar_decoders[i + 1](refined[i], scene_features, ego_status_last)
            )
        parents = stages[-1]
        return dict(proposals=parents, proposal_list=stages)
