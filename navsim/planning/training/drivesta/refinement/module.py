from navsim.common.compatibility.checkpoints import normalize_checkpoint
import csv
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch


class DriveSTALightningModule(pl.LightningModule):
    def __init__(
        self,
        agent,
        dataset_size,
        validation=None,
        validation_oracle=False,
        training_split="train",
        replay_samples=64,
        online=None,
        target_policy=None,
        data_protocol=None,
    ):
        super().__init__()
        self.agent = agent
        self.dataset_size = dataset_size
        self.validation = validation
        self.validation_oracle = validation_oracle
        self._resume_metadata = None
        self._pending_rng = None
        self.training_split = training_split
        self.replay_samples = replay_samples
        self.online = online
        self.target_policy = target_policy
        self._last_recovery_step = 0
        self.data_protocol = data_protocol

    def supervision_protocol(self):
        cfg = self.agent._config
        keys = (
            "training_phase",
            "jepa_checkpoint",
            "wta_targets_per_parent",
            "pair_max_ade",
            "pair_max_fde",
            "pair_max_heading",
            "diffusion_delta_scale",
            "diffusion_eval_noise_seed",
            "ddim_num_train_timesteps",
            "ddim_beta_schedule",
            "diffusion_num_inference_steps",
            "jepa_weight_decay",
            "jepa_encoder_lr",
            "jepa_predictor_lr",
            "jepa_layer_decay",
            "jepa_ema_momentum",
            "jepa_predictor_dim",
            "jepa_predictor_layers",
            "jepa_predictor_heads",
            "jepa_mask_tokens",
            "jepa_video_protocol",
            "jepa_activation_checkpointing",
        )
        joint = str(cfg.training_phase) == "joint"
        return dict(
            loss=self.agent.loss.protocol(),
            replay_samples=self.replay_samples,
            sampling="epoch_rotating_24_high_24_mid_8_low_8_zero_nonzero_backfill",
            target_policy=self.target_policy,
            data=self.data_protocol,
            online_labels="official_reference_plus_one",
            drop_last=False,
            refinement={k: getattr(cfg, k) for k in keys},
            retained_slot=(
                "none_all64_refined" if joint else "frozen_original_v2_selected"
            ),
            residual_prediction="v_prediction",
            frozen_observation_precision=(
                "jepa_only_fp32_trainable_scene_amp" if joint else "fp32"
            ),
            scorer_candidate_precision="fp32",
            joint_gradient_policy=(
                "detached_AR_residual_labels_trainable_scene" if joint else None
            ),
            jepa_input="encoder_F0_previous_current_crop28_resize512x256_RGB_ImageNet",
            motion_input="four_history_ego_states_44D",
            jepa_gradient_policy="expert_future_only_detach_before_both_planning_adapters",
            future_protocol="drive_jepa_expert_video8_full14_v1_multiblock_masked_layernorm_L1",
            gradient_clipping="separate_planner_and_future_norms",
            optimizer=dict(self.agent._lr_args),
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.agent._drivesta_model.jepa.update_target()

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        if gradient_clip_algorithm not in ("norm", None):
            raise ValueError(
                "DriveSTA uses independent norm clipping for planning/future objectives"
            )
        if gradient_clip_val and gradient_clip_val > 0:
            clip_independent_domains(optimizer, gradient_clip_val)

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        features, targets = batch
        if (
            "sampling_epoch" in targets
            and not (targets["sampling_epoch"] == self.current_epoch).all()
        ):
            raise RuntimeError("Dataset sampling epoch differs from optimizer epoch")
        self.agent.loss.epoch_progress = self.current_epoch + batch_idx / max(
            1, self.trainer.num_training_batches
        )
        prediction = self.agent(features, targets)
        if self.agent._config.training_phase != "dit":
            labels = self.online.score(
                targets["token"], prediction["proposals"].detach().float().cpu().numpy()
            )
            targets["native_labels"] = torch.as_tensor(
                labels, device=prediction["proposals"].device, dtype=torch.float32
            )
        losses = self.agent.compute_loss(features, targets, prediction)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Nonfinite feedback training loss")
        size = targets["trajectory"].shape[0]
        for key, value in losses.items():
            visible = key in (
                "loss",
                "expert_loss",
                "native_score_loss",
                "native_pdm_mae",
                "native_selection_regret",
            )
            self.log(
                f"train/{key}",
                value,
                on_step=True,
                on_epoch=True,
                batch_size=size,
                sync_dist=True,
                prog_bar=visible,
            )
        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        loader = self.trainer.train_dataloader
        if hasattr(loader, "acknowledge"):
            loader.acknowledge(self.current_epoch, len(batch[1]["token"]))

    def on_train_batch_start(self, batch, batch_idx):
        # Model/DDP and DataLoader iterator initialization may draw random
        # numbers. Restore each rank's saved stream only after these operations.
        if self._pending_rng is not None:
            from navsim.planning.training.drivesta.rng_state import restore_rng_states

            restore_rng_states(self._pending_rng)
            self._pending_rng = None
        # Save between updates: the prior batch is fully committed in both the
        # optimizer and the resumable loader; prefetch does not advance its cursor.
        step = self.trainer.global_step
        if step > 0 and step % 1000 == 0 and step != self._last_recovery_step:
            destination = Path(self.trainer.default_root_dir) / "recovery" / "last.ckpt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.trainer.save_checkpoint(str(destination))
            self._last_recovery_step = step

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        pred = self.agent(features)
        if self.validation is None:
            raise RuntimeError("Validation needs complete official metric caches")
        proposals = pred["proposals"].float().cpu().numpy()
        choices = pred["pdm_score"].argmax(-1).cpu().tolist()
        predictions = pred["pdm_score"].float().cpu().numpy()
        for token, bank, choice, scores in zip(
            targets["token"], proposals, choices, predictions
        ):
            labels = self.validation.score(
                token, bank if self.validation_oracle else bank[choice : choice + 1]
            )
            selected = float(labels[choice if self.validation_oracle else 0, 6] * 100)
            row = {"token": token, "pdms": selected}
            if self.validation_oracle:
                row.update(oracle=float(labels[:, 6].max() * 100))
                row["regret"] = row["oracle"] - selected
                row["candidate_high_fraction"] = float((labels[:, 6] >= 0.95).mean())
                row["candidate_mean_pdms"] = float(labels[:, 6].mean() * 100)
                row["scorer_mae_points"] = float(
                    np.abs(scores - labels[:, 6]).mean() * 100
                )
            self.validation_rows.append(row)

    def on_validation_epoch_end(self):
        rows = self.validation_rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for shard in gathered for row in shard]
        # DistributedSampler may pad the final batch. Count each token once.
        rows = sorted(
            {row["token"]: row for row in rows}.values(), key=lambda r: r["token"]
        )
        if not rows:
            raise ValueError("Validation produced no scenes")
        for key in rows[0]:
            if key != "token":
                self.log(
                    f"val/{key}",
                    float(np.mean([r[key] for r in rows])),
                    sync_dist=False,
                )
        self.log("val/scenes", float(len(rows)), sync_dist=False)
        if self.trainer.is_global_zero:
            root = Path(self.trainer.default_root_dir)
            root.mkdir(parents=True, exist_ok=True)
            with (root / f"val_epoch_{self.current_epoch:03d}.csv").open(
                "w", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def teardown(self, stage):
        if self.online is not None:
            self.online.close()

    def configure_optimizers(self):
        if int(self.agent.scheduler_args.dataset_size) != self.dataset_size:
            raise ValueError("DATA_SIZE does not match the filtered training dataset")
        self.agent.set_training_schedule(
            self.trainer.estimated_stepping_batches,
            self.trainer.world_size,
            self.trainer.accumulate_grad_batches,
        )
        return self.agent.get_optimizers()

    def on_fit_start(self):
        schedule = {
            "dataset_size": self.dataset_size,
            "training_split": self.training_split,
            "batch_per_gpu": self.agent.batch_size,
            "world_size": self.trainer.world_size,
            "accumulation": self.trainer.accumulate_grad_batches,
            "total_steps": int(self.trainer.estimated_stepping_batches),
            "supervision": self.supervision_protocol(),
            "refinement_phase": str(self.agent._config.training_phase),
        }
        if self._resume_metadata:
            for key, value in schedule.items():
                if key in self._resume_metadata and self._resume_metadata[key] != value:
                    raise ValueError(
                        f"Resume changes {key}; use checkpoint initialization for a new training schedule"
                    )
        if self.trainer.is_global_zero:
            root = Path(self.trainer.default_root_dir)
            root.mkdir(parents=True, exist_ok=True)
            optimizer = self.trainer.optimizers[0]
            schedule["peak_lr"] = optimizer.param_groups[0]["initial_lr"]
            schedule["optimizer_groups"] = {
                g["name"]: dict(
                    peak_lr=g["initial_lr"],
                    parameters=sum(p.numel() for p in g["params"]),
                    domain=g["domain"],
                )
                for g in optimizer.param_groups
            }
            (root / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")

    def on_save_checkpoint(self, checkpoint):
        from navsim.planning.training.drivesta.rng_state import capture_rng_states

        # Lightning invokes this hook on every rank before rank-zero writing.
        checkpoint["feedback_rng"] = capture_rng_states()
        dit_trained = (
            str(self.agent._config.training_phase) in ("dit", "joint")
            and self.trainer.global_step > 0
        )
        dit_trained |= bool(
            getattr(self.agent, "feedback_checkpoint_training", {}).get(
                "refinement_dit_trained", False
            )
        )
        dit_trained |= bool(
            (self._resume_metadata or {}).get("refinement_dit_trained", False)
        )
        checkpoint["feedback_training"] = {
            "dataset_size": self.dataset_size,
            "training_split": self.training_split,
            "total_steps": int(self.trainer.estimated_stepping_batches),
            "batch_per_gpu": self.agent.batch_size,
            "world_size": self.trainer.world_size,
            "accumulation": self.trainer.accumulate_grad_batches,
            "loss": type(self.agent.loss).__name__,
            "supervision": self.supervision_protocol(),
            "refinement_phase": str(self.agent._config.training_phase),
            "refinement_dit_trained": dit_trained,
        }

    def on_load_checkpoint(self, checkpoint):
        normalize_checkpoint(checkpoint)
        metadata = checkpoint.get("feedback_training")
        if metadata is None:
            raise ValueError(
                "Legacy checkpoints are initialization inputs; use agent.checkpoint_path"
            )
        if metadata.get("loss") != "DriveSTALoss":
            raise ValueError(
                "Refinement optimizer resume requires Refinement weights; initialize from Proposal separately"
            )
        if metadata["dataset_size"] != self.dataset_size:
            raise ValueError("Resume dataset size differs from checkpoint")
        if metadata.get("supervision") != self.supervision_protocol():
            raise ValueError(
                "Resume changes feedback supervision; use checkpoint initialization for a new loss or replay"
            )
        if (
            metadata.get("training_split", "train") == "trainval"
            and self.validation is not None
        ):
            raise ValueError(
                "A trainval-trained checkpoint cannot use overlapping val scenes for model selection"
            )
        if "feedback_rng" not in checkpoint:
            raise ValueError(
                "Full Refinement resume needs per-rank RNG state; use weight initialization for older checkpoints"
            )
        self._pending_rng = checkpoint["feedback_rng"]
        self._resume_metadata = metadata


def clip_independent_domains(optimizer, maximum):
    # A large dense/scoring gradient must not rescale the JEPA update through a
    # shared global clipping norm. Both groups still use the same norm limit.
    for domain in ("planner", "future"):
        parameters = [
            p
            for g in optimizer.param_groups
            if g["domain"] == domain
            for p in g["params"]
            if p.grad is not None
        ]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True)
