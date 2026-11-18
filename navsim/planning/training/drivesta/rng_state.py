"""Rank-specific stochastic state for exact continuation of joint training."""

import random

import numpy as np
import torch


RNG_PROTOCOL = "python_numpy_torch_cpu_current_cuda_per_rank_v1"


def _distributed_context():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0


def capture_rng_states():
    """Call on EVERY rank from the checkpoint hook, never only global rank zero.

    The checkpoint contains all ranks, since Lightning writes a single shared
    file. Capture only an already initialized current CUDA device; inspecting a
    CPU run must not initialize CUDA or advance another device's generator.
    """
    world_size, rank = _distributed_context()
    local = dict(
        rank=rank,
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=(
            torch.cuda.get_rng_state(torch.cuda.current_device()).cpu().clone()
            if torch.cuda.is_initialized()
            else None
        ),
    )
    states = [None] * world_size
    if world_size > 1:
        torch.distributed.all_gather_object(states, local)
    else:
        states[0] = local
    return dict(protocol=RNG_PROTOCOL, world_size=world_size, rank_states=states)


def restore_rng_states(saved):
    """Restore this rank immediately before the first resumed training forward.

    Defer this until on_train_batch_start: constructing the model and the data
    iterator can consume RNG after Lightning's on_load_checkpoint hook. Restore
    once, after those operations, and before dropout/DiT noise are sampled.
    CUDA indices are process-local, so restore to the current device rather
    than the device index used by the process that produced the checkpoint.
    """
    world_size, rank = _distributed_context()
    if not isinstance(saved, dict) or saved.get("protocol") != RNG_PROTOCOL:
        raise ValueError("Checkpoint lacks the complete V4 RNG protocol")
    if saved.get("world_size") != world_size:
        raise ValueError("Resume changes RNG world size")
    states = saved.get("rank_states")
    if (
        not isinstance(states, list)
        or len(states) != world_size
        or any(
            not isinstance(state, dict) or state.get("rank") != i
            for i, state in enumerate(states)
        )
    ):
        raise ValueError("Checkpoint RNG ranks are incomplete or reordered")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if any(not required.issubset(state) for state in states):
        raise ValueError("Checkpoint has an incomplete RNG state")
    local = states[rank]
    has_cuda = torch.cuda.is_initialized()
    if (local["torch_cuda"] is not None) != has_cuda:
        raise ValueError("Resume changes RNG CUDA device availability")
    random.setstate(local["python"])
    np.random.set_state(local["numpy"])
    torch.set_rng_state(local["torch_cpu"].cpu())
    if has_cuda:
        torch.cuda.set_rng_state(local["torch_cuda"].cpu(), torch.cuda.current_device())
