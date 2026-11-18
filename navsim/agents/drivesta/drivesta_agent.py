"""DriveSTA agent using saved dense feedback for training."""

from navsim.common.compatibility.checkpoints import normalize_checkpoint

import math

import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.agents.drivesta.builders.feature_builder import DriveSTAFeatureBuilder
from navsim.agents.drivesta.builders.target_builder import DriveSTATargetBuilder
from navsim.planning.training.drivesta.refinement.callbacks import LitProgressBar
from navsim.agents.drivesta.drivesta_model import DriveSTAModel, CAMERA_CONFIG_KEYS


def lr_multiplier(step, total_steps, warmup_fraction=0.1):
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    warmup = min(max(1, int(total_steps * warmup_fraction)), max(1, total_steps - 1))
    if step < warmup:
        return 1e-6 + (1 - 1e-6) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


class DriveSTAAgent(AbstractAgent):
    def __init__(
        self,
        config,
        lr_args,
        checkpoint_path=None,
        checkpoint_strict_load=False,
        loss=None,
        progress_bar=True,
        console_log_every_n_steps=100,
        scheduler_args=None,
        batch_size=64,
        num_gpus=1,
    ):
        super().__init__()
        if any(
            bool(getattr(config, k, False))
            for k in (
                "use_online_vlm",
                "use_vlm_feature_cache",
                "scorer_use_serial_vlm_context",
            )
        ):
            raise ValueError(
                "This student accepts current observations only; VLM inputs are disabled"
            )
        self._config = config
        official_weights = dict(noc=1.0, dac=1.0, ddc=0.0, ttc=5.0, ep=5.0, comfort=2.0)
        if any(
            float(getattr(config, key)) != value
            for key, value in official_weights.items()
        ):
            raise ValueError(
                "Refinement scoring weights must match the official PDM loss: "
                + str(official_weights)
            )
        if config.use_diffusion_decoder and str(config.training_phase) == "joint" and (
            not config.use_diffusion_decoder
            or config.train_diffusion_decoder_only
            or not config.train_scorer
        ):
            raise ValueError(
                "Joint Refinement requires DiT and scorer enabled, with train_diffusion_decoder_only=false"
            )
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self._checkpoint_strict_load = checkpoint_strict_load
        self.progress_bar = progress_bar
        self.console_log_every_n_steps = console_log_every_n_steps
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        if checkpoint_strict_load and config.use_diffusion_decoder:
            # Complete Refinement checkpoints already contain online/EMA/predictor.
            config.jepa_load_pretrained = False
        if checkpoint_strict_load:
            from omegaconf import open_dict
            with open_dict(config):
                config.image_backbone.load_pretrained = False
                config.lidar_backbone.load_pretrained = False
        # Expose the full model under the DriveSTA namespace.
        self._drivesta_model = DriveSTAModel(config)
        self.loss = loss
        self._training_schedule = None

    def name(self):
        return self.__class__.__name__

    def initialize(self):
        if self._checkpoint_path:
            self._load_pretrained_checkpoint()

    def get_sensor_config(self):
        return SensorConfig(
            **{
                key: OmegaConf.to_object(self._config[key])
                for key in CAMERA_CONFIG_KEYS + ("lidar_pc",)
            }
        )

    def get_feature_builders(self):
        return [DriveSTAFeatureBuilder(self._config)]

    def get_target_builders(self):
        return [DriveSTATargetBuilder(self._config)]

    def forward(self, features, targets=None):
        if self.training and targets is None:
            raise ValueError("Training targets are required")
        return self._drivesta_model(features, targets if self.training else None)

    def compute_loss(self, features, targets, pred):
        return self.loss(targets, pred)

    @torch.no_grad()
    def compute_trajectory(self, agent_input):
        from navsim.common.dataclasses import Trajectory

        if not agent_input.token:
            raise ValueError(
                "A current scene token is required for reproducible DDIM sampling"
            )
        self.eval()
        features = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        device = next(self.parameters()).device
        features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
        features["scenario_token"] = [agent_input.token]
        return Trajectory(self(features)["trajectory"][0].float().cpu().numpy())

    def get_training_callbacks(self):
        callbacks = [
            ModelCheckpoint(
                save_top_k=-1,
                every_n_epochs=1,
                save_last=True,
                filename="epoch-{epoch:02d}-step-{step}",
            ),
            LearningRateMonitor(
                logging_interval="step", log_momentum=False, log_weight_decay=False
            ),
        ]
        if not self.progress_bar:
            callbacks.insert(
                1, LitProgressBar(log_every_n_steps=self.console_log_every_n_steps)
            )
        return callbacks

    def _load_pretrained_checkpoint(self):
        if not self._config.use_diffusion_decoder:
            return self._load_proposal_checkpoint()
        state = self._load_checkpoint_state()
        metadata = self.feedback_checkpoint_training
        source = metadata.get("loss")
        saved = metadata.get("supervision", {})
        if source == "DriveSTALoss":
            if metadata.get("refinement_phase") != "joint" or not metadata.get(
                "refinement_dit_trained", False
            ):
                raise ValueError(
                    "DriveSTA evaluation requires a trained joint Refinement checkpoint"
                )
            keys = (
                "diffusion_delta_scale",
                "ddim_num_train_timesteps",
                "ddim_beta_schedule",
                "diffusion_num_inference_steps",
                "diffusion_eval_noise_seed",
            )
            changed = [
                key
                for key in keys
                if saved.get("refinement", {}).get(key) != getattr(self._config, key)
            ]
            if changed:
                raise ValueError(
                    "Checkpoint generator protocol mismatch: " + ", ".join(changed)
                )
            if (
                saved.get("jepa_gradient_policy")
                != "expert_future_only_detach_before_both_planning_adapters"
            ):
                raise ValueError(
                    "Checkpoint uses an incompatible JEPA gradient protocol"
                )
            if (
                saved.get("future_protocol")
                != "drive_jepa_expert_video8_full14_v1_multiblock_masked_layernorm_L1"
            ):
                raise ValueError(
                    "Checkpoint uses an incompatible JEPA auxiliary protocol"
                )
            self.load_state_dict(state, strict=True)
            return
        if self._checkpoint_strict_load:
            raise ValueError(
                "Strict evaluation requires a complete DriveSTA Refinement checkpoint"
            )
        if source != "DriveSTAProposalLoss":
            raise ValueError(
                "Initialize DriveSTA from its Proposal or complete Refinement checkpoint"
            )
        if not self._config.jepa_load_pretrained:
            raise ValueError(
                "Proposal initialization requires pretrained JEPA encoder, predictor and EMA"
            )
        if any(".diffusion_decoder." in key for key in state):
            raise ValueError(
                "Proposal initialization unexpectedly contains a residual decoder"
            )
        prefix = "_drivesta_model."
        new = (
            "trajectory_decoder.diffusion_decoder.",
            "jepa.",
            "dit_condition.",
            "score_condition.",
            "reference_scorer.",
            "score_decoder.jepa_",
        )
        result = self.load_state_dict(state, strict=False)
        missing = [
            key
            for key in result.missing_keys
            if not any(key.startswith(prefix + name) for name in new)
        ]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"Initialization mismatch: {missing[:8]}, {result.unexpected_keys[:8]}"
            )
        model = self._drivesta_model
        model.reference_scorer.load_state_dict(
            model.score_decoder.state_dict(), strict=True
        )

    def _load_checkpoint_state(self):
        # Weight initialization/evaluation does not need eager optimizer-state IO.
        checkpoint = torch.load(
            self._checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("state_dict"), dict
        ):
            raise ValueError("Expected a Lightning checkpoint with state_dict")
        normalize_checkpoint(checkpoint)
        self.feedback_checkpoint_training = checkpoint.get("feedback_training", {})
        state = checkpoint["state_dict"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, "agent.")
        return state

    def set_training_schedule(self, total_steps, world_size, accumulate_grad_batches):
        self._training_schedule = (
            int(total_steps),
            int(world_size),
            int(accumulate_grad_batches),
        )

    def get_optimizers(self):
        if self._training_schedule is None:
            raise RuntimeError(
                "Use the feedback training entry: scheduler needs the actual optimizer-step count"
            )
        total_steps, world_size, accumulation = self._training_schedule
        effective_batch = self.batch_size * world_size * accumulation
        lr = self._lr_args["base_lr"] * math.sqrt(
            effective_batch / self._lr_args["base_batch_size"]
        )
        optimizer_type = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[
            self._lr_args["name"]
        ]
        groups = (optimizer_groups(self, lr) if self._config.use_diffusion_decoder
                  else (p for p in self.parameters() if p.requires_grad))
        optimizer = optimizer_type(groups, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_multiplier(step, total_steps)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


    def _load_proposal_checkpoint(self):
        state = self._load_checkpoint_state()
        result = self.load_state_dict(state, strict=False)
        # Loading proposal into refinement may introduce only the new DiT parameters.
        allowed = "_drivesta_model.trajectory_decoder.diffusion_decoder."
        missing = [key for key in result.missing_keys if not key.startswith(allowed)]
        new_dit = {key for key in self.state_dict() if key.startswith(allowed)}
        missing_dit = set(result.missing_keys) & new_dit
        if missing_dit and missing_dit != new_dit:
            raise RuntimeError(
                "Partially missing DiT checkpoint; expected complete proposal or refinement weights"
            )
        if self._checkpoint_strict_load:
            missing = result.missing_keys
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing[:8]}, unexpected={result.unexpected_keys[:8]}"
            )



def optimizer_groups(agent, planner_lr):
    """Complete, disjoint groups. Only online JEPA/predictor belong to future loss."""
    cfg = agent._config
    prefix = "_drivesta_model.jepa."
    depth = len(agent._drivesta_model.jepa.encoder.blocks)
    grouped = {
        "planner": dict(params=[], lr=planner_lr, name="planner", domain="planner")
    }
    for name, parameter in agent.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = "bias" in name or parameter.ndim == 1
        wd = 0.0 if no_decay else float(cfg.jepa_weight_decay)
        suffix = "_no_decay" if no_decay else "_decay"
        if name.startswith(prefix + "encoder."):
            local = name[len(prefix + "encoder.") :]
            layer = (
                int(local.split(".")[1]) + 1
                if local.startswith("blocks.")
                else (depth + 1 if local.startswith("norm.") else 0)
            )
            key = f"jepa_encoder_layer{layer:02d}" + suffix
            lr = float(cfg.jepa_encoder_lr) * float(cfg.jepa_layer_decay) ** (
                depth + 1 - layer
            )
            group = grouped.setdefault(
                key, dict(params=[], lr=lr, name=key, domain="future", weight_decay=wd)
            )
        elif name.startswith(prefix + "predictor."):
            group = grouped.setdefault(
                "jepa_predictor" + suffix,
                dict(
                    params=[],
                    lr=float(cfg.jepa_predictor_lr),
                    name="jepa_predictor" + suffix,
                    domain="future",
                    weight_decay=wd,
                ),
            )
        elif name.startswith(prefix):
            raise ValueError(
                "EMA or unknown JEPA parameter must not be trainable: " + name
            )
        else:
            group = grouped["planner"]
        group["params"].append(parameter)
    groups = [
        grouped[k]
        for k in ["planner"] + sorted(k for k in grouped if k != "planner")
        if grouped[k]["params"]
    ]
    ids = [id(p) for g in groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {
        id(p) for p in agent.parameters() if p.requires_grad
    }:
        raise RuntimeError(
            "Optimizer groups do not cover trainable parameters exactly once"
        )
    return groups
