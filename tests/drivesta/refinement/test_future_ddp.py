"""Uneven missing-future masks must not bias JEPA gradients or EMA across ranks."""

from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tests.drivesta.refinement.test_future import make


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.jepa = make()[0]
        self.planner = nn.Linear(1024, 1)

    def forward(self, clip, ego, targets):
        context = self.jepa(clip)
        future = self.jepa.future_objective(targets)["loss"]
        return future + self.planner(context.detach()).square().mean()


def worker(rank, init, path):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=2
    )
    model = SmallModel()
    wrapped = DistributedDataParallel(model, find_unused_parameters=True)
    _, clip, ego, targets = make()
    clip = clip + rank * 0.1
    if rank == 0:
        targets["jepa_future_valid"].zero_()
    wrapped(clip, ego, targets).backward()
    gradient = model.jepa.encoder.blocks[0].weight.grad.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    model.jepa.update_target()
    before = {k: v.clone() for k, v in model.jepa.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    targets["jepa_future_valid"].zero_()
    wrapped(clip, ego, targets).backward()
    assert all(p.grad is None for p in model.jepa.encoder.parameters())
    optimizer.step()
    model.jepa.update_target()
    assert all(torch.equal(v, model.jepa.state_dict()[k]) for k, v in before.items())
    torch.save(
        dict(
            gradient=gradient,
            ema=model.jepa.target_encoder.blocks[0].weight.detach(),
            updates=int(model.jepa.ema_updates),
        ),
        Path(path) / f"rank{rank}.pt",
    )
    torch.distributed.destroy_process_group()


def test_ddp_one_empty_rank_matches_global_valid_mean(tmp_path):
    torch.multiprocessing.spawn(
        worker,
        args=("file://" + str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    first = torch.load(tmp_path / "rank0.pt", weights_only=True)
    second = torch.load(tmp_path / "rank1.pt", weights_only=True)
    assert (
        torch.equal(first["gradient"], second["gradient"])
        and first["updates"] == second["updates"] == 1
    )
    assert torch.equal(first["ema"], second["ema"])
    model = SmallModel()
    _, clip, ego, targets = make()
    # Only rank1 has a valid future; reference is precisely its valid-example mean.
    context = model.jepa(clip + 0.1)
    model.jepa.future_objective(targets)["loss"].backward()
    assert torch.allclose(
        first["gradient"],
        model.jepa.encoder.blocks[0].weight.grad,
        atol=1e-7,
        rtol=1e-5,
    )
