"""Epoch-addressed sampling, independent of worker count and prefetch order."""

import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


class EpochDistributedSampler(DistributedSampler):
    resume_epoch = -1
    resume_offset = 0

    def __iter__(self):
        # Lightning calls set_epoch before iterating. The epoch travels with each
        # index, so prefetched/persistent workers cannot observe a stale epoch.
        indices = list(super().__iter__())
        start = self.resume_offset if self.epoch == self.resume_epoch else 0
        return iter((index, self.epoch) for index in indices[start:])


class EpochDataLoader(DataLoader):
    """Save consumed records, not prefetched records, in Lightning checkpoints."""

    consumed_epoch = -1
    consumed_samples = 0

    def acknowledge(self, epoch, samples):
        if epoch != self.consumed_epoch:
            self.consumed_epoch, self.consumed_samples = epoch, 0
        self.consumed_samples += samples

    def state_dict(self):
        epoch = max(0, self.consumed_epoch)
        samples = self.consumed_samples
        if samples >= len(self.sampler):
            epoch, samples = epoch + 1, 0
        return dict(
            epoch=epoch,
            samples=samples,
            dataset_size=len(self.dataset),
            world_size=self.sampler.num_replicas,
            batch_size=self.batch_size,
            seed=self.sampler.seed,
        )

    def load_state_dict(self, state):
        current = self.state_dict()
        for key in ("dataset_size", "world_size", "batch_size", "seed"):
            if current[key] != state[key]:
                raise ValueError("Resume changes loader " + key)
        self.consumed_epoch, self.consumed_samples = state["epoch"], state["samples"]
        self.sampler.resume_epoch, self.sampler.resume_offset = (
            state["epoch"],
            state["samples"],
        )
        self.sampler.set_epoch(state["epoch"])


def cyclic_indices(pool, count, epoch, token, seed=2, stream=0):
    pool = np.asarray(pool, dtype=np.int64)
    if epoch < 0 or count < 0 or len(np.unique(pool)) != len(pool):
        raise ValueError("Invalid rotation input")
    if not len(pool) or count == 0:
        return np.empty(0, dtype=np.int64)
    number = int(token, 16)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), number & 0xFFFFFFFF, number >> 32, int(stream)]
        )
    )
    order = rng.permutation(pool)
    count = min(int(count), len(order))
    offsets = (epoch * count + np.arange(count)) % len(order)
    return order[offsets]


def rotating_replay_indices(labels, epoch, token, seed=2):
    score = np.asarray(labels)[:, -1]
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Invalid replay score")
    groups = [
        np.flatnonzero(score >= 0.95),
        np.flatnonzero((score >= 0.8) & (score < 0.95)),
        np.flatnonzero((score > 0) & (score < 0.8)),
        np.flatnonzero(score == 0),
    ]
    chosen = np.concatenate(
        [
            cyclic_indices(pool, quota, epoch, token, seed, stream=i + 1)
            for i, (pool, quota) in enumerate(zip(groups, (24, 24, 8, 8)))
        ]
    )
    if len(chosen) < 64:
        remaining = np.setdiff1d(np.flatnonzero(score > 0), chosen)
        extra = cyclic_indices(
            remaining, 64 - len(chosen), epoch, token, seed, stream=5
        )
        chosen = np.concatenate([chosen, extra])
    if not len(chosen):
        raise ValueError("No labelled replay candidates")
    # Context order has its own deterministic permutation; selected membership
    # follows the fixed stratum rotations above.
    return cyclic_indices(chosen, len(chosen), 0, token, seed, stream=6 + epoch)
