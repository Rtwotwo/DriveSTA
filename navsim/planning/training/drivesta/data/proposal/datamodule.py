"""Construct rank-aware loaders only after Lightning initializes DDP."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from navsim.planning.training.drivesta.data.proposal.sampling import EpochDistributedSampler, EpochDataLoader


class DriveSTAProposalDataModule(pl.LightningDataModule):
    def __init__(self, train, val, loader_params, collate_fn, seed=2):
        super().__init__()
        self.train = train
        self.val = val
        self.loader_params = loader_params
        self.collate_fn = collate_fn
        self.seed = int(seed)
        self.resume_loader_state = None

    def train_dataloader(self):
        sampler = EpochDistributedSampler(
            self.train,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=True,
            seed=self.seed,
            drop_last=False,
        )
        loader = EpochDataLoader(
            self.train,
            sampler=sampler,
            drop_last=False,
            collate_fn=self.collate_fn,
            **self.loader_params,
        )
        # estimated_stepping_batches can prefetch before Lightning restores loops.
        # Install the saved consumed cursor before the FIRST iterator is created.
        if self.resume_loader_state is not None:
            loader.load_state_dict(self.resume_loader_state)
            self.resume_loader_state = None
        return loader

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
