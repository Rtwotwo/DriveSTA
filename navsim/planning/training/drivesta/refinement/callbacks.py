# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# Modified from NAVSIM (https://github.com/autonomousvision/navsim)
# ------------------------------------------------------------------------
"""Console progress reporting for dense trajectory training."""
import time
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar


class LitProgressBar(ProgressBar):

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.log_every_n_steps = log_every_n_steps
        self._train_epoch_start_t = None
        self._val_epoch_start_t = None

    def disable(self):
        self.enable = False

    @staticmethod
    def _is_rank_zero(trainer) -> bool:
        # Lightning provides is_global_zero in most versions
        if hasattr(trainer, "is_global_zero"):
            try:
                return bool(trainer.is_global_zero)
            except Exception:
                pass
        # Fallbacks
        for attr in ("global_rank", "local_rank", "node_rank"):
            if hasattr(trainer, attr):
                try:
                    return int(getattr(trainer, attr)) == 0
                except Exception:
                    continue
        return True

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds_i = max(0, int(seconds))
        h = seconds_i // 3600
        m = (seconds_i % 3600) // 60
        s = seconds_i % 60
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    @staticmethod
    def _now_ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._train_epoch_start_t = time.monotonic()
        return super().on_train_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._val_epoch_start_t = time.monotonic()
        return super().on_validation_epoch_start(trainer, pl_module)

    def _format_line(
        self, phase: str, epoch: int, batch_idx: int, total_batches: int, start_t
    ) -> str:
        elapsed_s = None
        eta_s = None
        if start_t is not None and batch_idx >= 0:
            elapsed_s = time.monotonic() - start_t
            done = max(1, batch_idx + 1)
            remain = max(0, total_batches - done)
            eta_s = (elapsed_s / done) * remain

        parts = [
            f"[{self._now_ts()}] Epoch {epoch} - {phase} {batch_idx} / {total_batches}"
        ]
        if elapsed_s is not None:
            parts.append(f"elapsed {self._fmt_duration(elapsed_s)}")
        if eta_s is not None:
            parts.append(f"ETA {self._fmt_duration(eta_s)}")
        return " - ".join(parts)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if not self._is_rank_zero(trainer):
            return
        if batch_idx % self.log_every_n_steps == 0:
            total = int(
                getattr(self, "total_train_batches", None)
                or getattr(trainer, "num_training_batches", 0)
                or 0
            )
            total = total if total > 0 else (batch_idx + 1)
            prefix = self._format_line(
                "train",
                trainer.current_epoch,
                batch_idx,
                total,
                self._train_epoch_start_t,
            )
            print(f"{prefix} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if not self._is_rank_zero(trainer):
            return
        if batch_idx % self.log_every_n_steps == 0:
            total = int(getattr(self, "total_val_batches", None) or 0)
            if total <= 0:
                total = batch_idx + 1
            prefix = self._format_line(
                "val", trainer.current_epoch, batch_idx, total, self._val_epoch_start_t
            )
            print(f"{prefix} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if not self._is_rank_zero(trainer):
            return
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k, v in metrics.items():
            if "train/" in k:
                train_metrics[k] = v
            elif "val/" in k:
                val_metrics[k] = v
            else:
                other_metrics[k] = v
        print(
            f"\n[{self._now_ts()}] ###########  Epoch {trainer.current_epoch} ##########"
        )
        for k, v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k, v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k, v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")
