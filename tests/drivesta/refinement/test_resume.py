"""Exercise consumed-data recovery with real Lightning, including prefetch."""

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from navsim.planning.training.drivesta.data.refinement.sampling import (
    EpochDistributedSampler,
    EpochDataLoader,
)


class Indexed(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, key):
        return key


class Interrupted(Exception):
    pass


class CursorModel(pl.LightningModule):
    def __init__(self, stop_at=None, destination=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.seen = []
        self.stop_at = stop_at
        self.destination = destination

    def train_dataloader(self):
        data = Indexed()
        return EpochDataLoader(
            data,
            batch_size=3,
            num_workers=2,
            prefetch_factor=2,
            sampler=EpochDistributedSampler(
                data, num_replicas=1, rank=0, shuffle=True, seed=2
            ),
        )

    def training_step(self, batch, batch_idx):
        ids, epochs = batch
        assert (epochs == self.current_epoch).all()
        self.seen.extend((self.current_epoch, int(i)) for i in ids)
        return (self.weight - ids.float() / 12).square().mean()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.trainer.train_dataloader.acknowledge(self.current_epoch, len(batch[0]))

    def on_train_batch_start(self, batch, batch_idx):
        if self.stop_at is not None and self.global_step == self.stop_at:
            self.trainer.save_checkpoint(str(self.destination))
            raise Interrupted()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)


def trainer(root, callbacks=None):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(root),
        callbacks=callbacks or [],
        use_distributed_sampler=False,
    )


@pytest.mark.parametrize("boundary", ["mid_epoch", "epoch_end"])
def test_epoch_training_resume_preserves_token_sequence_and_optimizer(
    tmp_path, boundary
):
    reference = CursorModel()
    saver = pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    trainer(tmp_path / "reference", [saver]).fit(reference)
    if boundary == "mid_epoch":
        checkpoint = tmp_path / "interrupted.ckpt"
        original = CursorModel(stop_at=2, destination=checkpoint)
        with pytest.raises(Interrupted):
            trainer(tmp_path / "interrupted").fit(original)
        prefix = original.seen
    else:
        checkpoint = next(
            (tmp_path / "reference" / "checkpoints").glob("epoch=0-*.ckpt")
        )
        prefix = reference.seen[:12]
    restored = CursorModel()
    trainer(tmp_path / "restored").fit(restored, ckpt_path=str(checkpoint))
    assert prefix + restored.seen == reference.seen
    assert torch.equal(restored.weight, reference.weight)
