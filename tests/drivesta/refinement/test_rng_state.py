"""Stochastic optimizer continuation and distinct per-rank RNG restoration."""

import random

import numpy as np
import pytest
import torch
from torch import nn

from navsim.planning.training.drivesta.rng_state import (
    capture_rng_states,
    restore_rng_states,
)


def seed_streams(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def draw_streams():
    return random.random(), np.random.rand(3), torch.rand(3)


def assert_same_draw(first, second):
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_stochastic_adamw_resume_matches_uninterrupted_after_iterator_creation(
    tmp_path,
):
    seed_streams(271)
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.3), nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    def step(network, opt):
        opt.zero_grad(set_to_none=True)
        # Exercise the sources used by augmentation, dropout, and DiT t/noise.
        x = torch.randn(3, 4) + torch.randint(1000, (3, 1)) / 1000
        loss = (network(x) - random.random() - np.random.rand()).square().mean()
        loss.backward()
        opt.step()
        return loss.detach().clone()

    step(model, optimizer)
    path = tmp_path / "stochastic.ckpt"
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=capture_rng_states(),
        ),
        path,
    )
    expected_losses = [step(model, optimizer) for _ in range(3)]

    # Reinitialization and DataLoader iterator creation consume torch RNG.
    seed_streams(271)
    resumed = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.3), nn.Linear(8, 1))
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    iterator = iter(torch.utils.data.DataLoader(torch.arange(8), batch_size=2))
    next(iterator)
    restore_rng_states(checkpoint["rng"])
    actual_losses = [step(resumed, resumed_optimizer) for _ in range(3)]
    assert all(torch.equal(a, b) for a, b in zip(expected_losses, actual_losses))
    assert all(
        torch.equal(value, resumed.state_dict()[key])
        for key, value in model.state_dict().items()
    )
    for first, second in zip(
        optimizer.state.values(), resumed_optimizer.state.values()
    ):
        assert all(torch.equal(value, second[key]) for key, value in first.items())


def test_incompatible_rng_world_size_is_rejected_before_restoring():
    seed_streams(41)
    state = capture_rng_states()
    expected = draw_streams()
    restore_rng_states(state)
    with pytest.raises(ValueError, match="world size"):
        restore_rng_states(dict(state, world_size=2))
    assert_same_draw(draw_streams(), expected)


def _rank_rng_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2
    )
    try:
        seed_streams(100 + rank)
        state = capture_rng_states()
        assert state["world_size"] == 2
        assert not torch.equal(
            state["rank_states"][0]["torch_cpu"], state["rank_states"][1]["torch_cpu"]
        )
        expected = draw_streams()
        seed_streams(999)  # Both ranks must still recover different saved streams.
        restore_rng_states(state)
        assert_same_draw(draw_streams(), expected)
        restore_rng_states(state)
        assert_same_draw(draw_streams(), expected)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_checkpoint_restores_each_ranks_own_stream(tmp_path):
    torch.multiprocessing.spawn(
        _rank_rng_worker,
        args=((tmp_path / "rng_init").resolve().as_uri(),),
        nprocs=2,
        join=True,
    )


def test_training_iterator_does_not_consume_model_rng():
    from types import SimpleNamespace
    from torch.utils.data import Dataset
    from navsim.planning.training.drivesta.data.refinement.datamodule import DriveSTADataModule

    class Small(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, key):
            return key[0]

    module = DriveSTADataModule(Small(), [], dict(batch_size=2, num_workers=0), None)
    module.trainer = SimpleNamespace(world_size=1, global_rank=0)
    loader = module.train_dataloader()
    before = torch.get_rng_state().clone()
    for epoch in range(2):
        loader.sampler.set_epoch(epoch)
        list(iter(loader))
    assert torch.equal(before, torch.get_rng_state())
