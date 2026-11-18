"""Actual Lightning interrupted-epoch recovery with prefetched workers."""

import tempfile
from pathlib import Path
import unittest
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from navsim.planning.training.drivesta.data.proposal.sampling import (
    EpochDataLoader,
    EpochDistributedSampler,
)


class TraceDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, key):
        return torch.tensor(key, dtype=torch.long)


class Interrupted(Exception):
    pass


class TraceModule(pl.LightningModule):
    def __init__(self, stop_path=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.0))
        self.trace = []
        self.stop_path = stop_path

    def training_step(self, batch, batch_idx):
        assert (batch[:, 1] == self.current_epoch).all()
        self.trace.extend([tuple(x) for x in batch.tolist()])
        return (self.w - batch[:, 0].float().mean() / 16).square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.trainer.train_dataloader.acknowledge(self.current_epoch, len(batch))

    def on_train_batch_start(self, batch, batch_idx):
        if self.stop_path and self.global_step == 2:
            self.trainer.save_checkpoint(self.stop_path)
            raise Interrupted()


class LoaderResumeTest(unittest.TestCase):
    def test_lightning_mid_epoch_resume_matches_trace_and_updates(self):
        def loader():
            ds = TraceDataset()
            return EpochDataLoader(
                ds,
                batch_size=4,
                num_workers=2,
                persistent_workers=True,
                sampler=EpochDistributedSampler(ds, num_replicas=1, rank=0, seed=2),
            )

        def trainer():
            return pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=2,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                use_distributed_sampler=False,
            )

        with tempfile.TemporaryDirectory() as d:
            full = TraceModule()
            trainer().fit(full, train_dataloaders=loader())
            path = str(Path(d) / "interrupted.ckpt")
            first = TraceModule(path)
            with self.assertRaises(Interrupted):
                trainer().fit(first, train_dataloaders=loader())
            second = TraceModule()
            resumed = trainer()
            resumed.fit(second, train_dataloaders=loader(), ckpt_path=path)
            self.assertEqual(first.trace + second.trace, full.trace)
            self.assertEqual(resumed.global_step, 8)
            torch.testing.assert_close(second.w, full.w, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
