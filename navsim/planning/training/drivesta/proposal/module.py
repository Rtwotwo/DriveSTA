from navsim.common.compatibility.checkpoints import normalize_checkpoint
import csv
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch


class DriveSTAProposalLightningModule(pl.LightningModule):
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
        self.training_split = training_split
        self.replay_samples = replay_samples
        self.online = online
        self.target_policy = target_policy
        self._last_recovery_step = 0
        self.data_protocol = data_protocol

    def supervision_protocol(self):
        return dict(
            loss=self.agent.loss.protocol(),
            replay_samples=self.replay_samples,
            sampling="epoch_rotating_24_high_24_mid_8_low_8_zero_nonzero_backfill",
            target_policy=self.target_policy,
            data=self.data_protocol,
            online_labels="official_reference_plus_one",
            drop_last=False,
        )

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
        labels = self.online.score(
            targets["token"],
            prediction.get("score_proposals", prediction["proposals"])
            .detach()
            .float()
            .cpu()
            .numpy(),
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
            (root / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["feedback_training"] = {
            "dataset_size": self.dataset_size,
            "training_split": self.training_split,
            "total_steps": int(self.trainer.estimated_stepping_batches),
            "batch_per_gpu": self.agent.batch_size,
            "world_size": self.trainer.world_size,
            "accumulation": self.trainer.accumulate_grad_batches,
            "loss": type(self.agent.loss).__name__,
            "supervision": self.supervision_protocol(),
        }

    def on_load_checkpoint(self, checkpoint):
        normalize_checkpoint(checkpoint)
        metadata = checkpoint.get("feedback_training")
        if metadata is None:
            raise ValueError(
                "Legacy checkpoints are initialization inputs; use agent.checkpoint_path"
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
        self._resume_metadata = metadata
