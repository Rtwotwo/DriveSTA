"""Construct rank-aware loaders only after Lightning initializes DDP."""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from navsim.planning.training.drivesta.data.refinement.sampling import EpochDistributedSampler, EpochDataLoader


class DriveSTADataModule(pl.LightningDataModule):
    def __init__(self, train, val, loader_params, collate_fn, seed=2):
        super().__init__()
        self.train = train
        self.val = val
        self.loader_params = loader_params
        self.collate_fn = collate_fn
        self.seed = int(seed)

    def train_dataloader(self):
        sampler = EpochDistributedSampler(
            self.train,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=True,
            seed=self.seed,
            drop_last=False,
        )
        return EpochDataLoader(
            self.train,
            sampler=sampler,
            drop_last=False,
            collate_fn=self.collate_fn,
            # Worker base-seed draws must not perturb DiT/dropout
            # RNG at epoch boundaries or during full restoration.
            generator=torch.Generator().manual_seed(
                self.seed + self.trainer.global_rank
            ),
            **self.loader_params,
        )

    def val_dataloader(self):
        sampler = DistributedSampler(
            self.val,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,
            drop_last=False,
        )
        return DataLoader(
            self.val,
            sampler=sampler,
            drop_last=False,
            collate_fn=self.collate_fn,
            **self.loader_params,
        )
