import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.agents.drivesta.encoders.jepa import TrainableJEPA, VideoCondition
from navsim.planning.training.drivesta.data.refinement.future import (
    load_future_index,
    future_target,
    FRAME_POLICY,
)
from navsim.planning.training.drivesta.refinement.module import clip_independent_domains


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 1024)])
        self.norm = nn.LayerNorm(1024)

    def forward(self, x, masks=None):
        from torch.nn import functional as F
        from navsim.agents.drivesta.encoders.jepa_vendor.masks.utils import apply_masks

        x = F.avg_pool3d(x, (2, 16, 16)).flatten(2).transpose(1, 2)
        if masks is not None:
            x = apply_masks(x, [masks])
        return self.norm(self.blocks[0](x))


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(1024, 1024)

    def forward(self, context, mx, my, mask_index):
        return self.project(context.mean(1, keepdim=True)).expand(-1, my.shape[1], -1)


def make():
    torch.manual_seed(9)
    cfg = SimpleNamespace(jepa_activation_checkpointing=False, jepa_ema_momentum=0.9)
    model = TrainableJEPA(
        None, cfg, encoder=TinyEncoder(), predictor=TinyPredictor()
    ).train()
    clip = torch.randn(2, 3, 2, 256, 512)
    targets = dict(
        jepa_future_valid=torch.tensor([True, False]),
        jepa_video_clip=torch.randn(2, 3, 8, 256, 512),
        jepa_video_offsets=torch.tensor(
            [[-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]] * 2
        ),
        jepa_mask_seed=torch.tensor([19, 31]),
    )
    return model, clip, torch.randn(2, 4, 11), targets


def test_future_updates_online_and_predictor_but_not_target():
    model, clip, ego, targets = make()
    assert model.encoder.training and not model.target_encoder.training
    result = model.future_objective(targets)
    result["loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.encoder.parameters()
    )
    assert sum(p.grad.abs().sum() for p in model.predictor.parameters()) > 0
    assert all(
        p.grad is None and not p.requires_grad
        for p in model.target_encoder.parameters()
    )


@pytest.mark.parametrize("pooled", [True, False])
def test_planning_adapters_learn_without_any_jepa_gradient(pooled):
    model, clip, ego, _ = make()
    adapter = VideoCondition(32, pooled)
    adapter(model(clip).detach(), ego).square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters()
    )
    assert all(p.grad is None for p in model.parameters())


def test_planner_gradient_scale_cannot_change_future_update_via_clipping():
    updates = []
    for multiplier in (1.0, 100000.0):
        model, clip, ego, targets = make()
        planner = nn.Linear(2, 1)
        parameters = list(model.encoder.parameters()) + list(
            model.predictor.parameters()
        )
        optimizer = torch.optim.AdamW(
            [
                dict(params=parameters, domain="future"),
                dict(params=planner.parameters(), domain="planner"),
            ],
            lr=0.001,
        )
        loss = model.future_objective(targets)["loss"]
        (loss + multiplier * planner(torch.ones(1, 2)).square().sum()).backward()
        clip_independent_domains(optimizer, 1.0)
        optimizer.step()
        model.update_target()
        updates.append({k: v.clone() for k, v in model.state_dict().items()})
    assert all(torch.equal(v, updates[1][k]) for k, v in updates[0].items())


def test_missing_future_no_decay_or_ema_update():
    model, clip, ego, targets = make()
    targets["jepa_future_valid"].zero_()
    targets["jepa_video_clip"].fill_(float("nan"))
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=0.1)
    result = model.future_objective(targets)
    assert result["loss"] == 0 and not result["loss"].requires_grad
    optimizer.step()
    model.update_target()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_ema_equation_and_checkpoint_roundtrip():
    model, clip, ego, targets = make()
    old = copy.deepcopy(model.target_encoder.state_dict())
    model.future_objective(targets)["loss"].backward()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.1)
    model.update_target()
    for k, p in model.target_encoder.named_parameters():
        assert torch.allclose(
            p, old[k] * 0.9 + model.encoder.state_dict()[k] * 0.1, atol=1e-6
        )
    restored = make()[0]
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.ema_updates == 1
    assert all(
        torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()
    )


def test_future_pixels_do_not_change_observed_context():
    model, clip, ego, targets = make()
    first = model(clip).detach().clone()
    model.future_objective(targets)
    targets["jepa_video_clip"].normal_(100, 10)
    model.future_objective(targets)
    assert torch.equal(first, model(clip))


def test_invalid_future_time_fails_closed():
    model, clip, ego, targets = make()
    targets["jepa_video_offsets"][0, 0] = -1.5
    with pytest.raises(ValueError, match="timestamps"):
        model.future_objective(targets)


def test_manifest_identity_and_missing_window(tmp_path):
    from navsim.planning.training.drivesta.data.refinement.future import assemble_video

    history = {"abc": dict(log_name="log", image_paths=["h", "c"], offsets=[-0.5, 0.0])}
    row = assemble_video(
        dict(
            log_name="log",
            history_paths=["h", "c"],
            history_offsets=[-0.5, 0.0],
            frames=[
                dict(
                    token="abc" if i == 3 else str(i),
                    offset=(i - 3) * 0.5,
                    image_path=str(i),
                    available=False,
                )
                for i in range(14)
            ],
        )
    )
    path = tmp_path / "manifest.json"
    payload = dict(
        frame_policy=FRAME_POLICY, source_identity="hash", records={"abc": row}
    )
    path.write_text(json.dumps(payload))
    rows, identity = load_future_index(
        path, [dict(token="abc", log_name="log")], "hash", history
    )
    assert identity == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not future_target(rows["abc"], "abc", 0)["jepa_future_valid"]
    with pytest.raises(ValueError, match="source/protocol"):
        load_future_index(path, [dict(token="abc", log_name="log")], "wrong", history)
    payload["records"]["abc"]["video_valid"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Misaligned"):
        load_future_index(path, [dict(token="abc", log_name="log")], "hash", history)


def test_full14_windows_rotate_without_fabricated_frames(tmp_path):
    import cv2, numpy as np
    from navsim.planning.training.drivesta.data.refinement.future import assemble_video

    frames = []
    for i in range(14):
        path = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(path), np.full((128, 256, 3), i * 10, np.uint8))
        frames.append(
            dict(
                token=str(i), offset=(i - 3) * 0.5, image_path=str(path), available=True
            )
        )
    row = assemble_video(
        dict(
            history_offsets=[-0.5, 0.0],
            history_paths=[f["image_path"] for f in frames[2:4]],
            frames=frames,
        )
    )
    samples = [future_target(row, "token", epoch) for epoch in range(7)]
    assert row["video_valid"] and len(row["video_clips"]) == 7
    assert {int(s["jepa_video_start"]) for s in samples} == set(range(7))
    assert torch.equal(
        samples[0]["jepa_video_clip"], future_target(row, "token", 0)["jepa_video_clip"]
    )
    for sample in samples:
        start = int(sample["jepa_video_start"])
        assert sample["jepa_video_offsets"].tolist() == [
            (i - 3) * 0.5 for i in range(start, start + 8)
        ]
    row["frames"][6]["available"] = False
    assert not assemble_video(row)[
        "video_valid"
    ]  # frame6 participates in every 8/14 window
    row["frames"][6]["available"] = True
    for f in row["frames"][7:]:
        f["offset"] += 0.5
    assert not assemble_video(row)[
        "video_valid"
    ]  # a central 1s gap cannot be interpolated


def test_adamw_scheduler_ema_resume_matches_uninterrupted(tmp_path):
    def setup():
        model, clip, ego, targets = make()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 6)
        return model, targets, optimizer, scheduler

    def step(model, targets, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        model.future_objective(targets)["loss"].backward()
        optimizer.step()
        model.update_target()
        scheduler.step()

    model, targets, opt, sched = setup()
    step(model, targets, opt, sched)
    file = tmp_path / "state.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
        ),
        file,
    )
    step(model, targets, opt, sched)
    other, t, o, s = setup()
    saved = torch.load(file, weights_only=True)
    other.load_state_dict(saved["model"])
    o.load_state_dict(saved["optimizer"])
    s.load_state_dict(saved["scheduler"])
    step(other, t, o, s)
    assert all(
        torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items()
    )
    assert sched.state_dict() == s.state_dict() and other.ema_updates == 2
    for first, second in zip(opt.state.values(), o.state.values()):
        assert all(torch.equal(value, second[key]) for key, value in first.items())


def test_future_optimizer_uses_official_weight_decay_exclusion():
    from navsim.agents.drivesta.drivesta_agent import optimizer_groups

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self._drivesta_model = nn.Module()
            self._drivesta_model.jepa = make()[0]
            self.planner = nn.Linear(3, 4)
            self._config = SimpleNamespace(
                jepa_encoder_lr=1e-5,
                jepa_predictor_lr=1e-4,
                jepa_layer_decay=0.9,
                jepa_weight_decay=0.04,
            )

    model = Agent()
    groups = optimizer_groups(model, 1.7e-4)
    names = {id(p): n for n, p in model.named_parameters()}
    for group in groups:
        if group["domain"] == "planner":
            assert "weight_decay" not in group
        else:
            for p in group["params"]:
                expected = 0.0 if "bias" in names[id(p)] or p.ndim == 1 else 0.04
                assert group["weight_decay"] == expected
