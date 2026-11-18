import json
from pathlib import Path
import numpy as np
import pytest

from drivesta_feedback.batch.simscale_recovery import gt_proposals, publish_stage
from drivesta_feedback.batch.physical_sampling import physical_mask, score_mask, select_bank
from drivesta_feedback.batch.filter_targets import TargetPolicy


def reference():
    t = np.arange(1, 9) * 0.5
    return np.column_stack([3 * t, np.zeros(8), np.zeros(8)]).astype(np.float32)


def test_zero_perturbation_preserves_recorded_reference():
    assert np.array_equal(gt_proposals(reference(), np.zeros((1, 6)))[0], reference())


def test_timing_shift_does_not_require_future_beyond_four_seconds():
    c = np.zeros((2, 6))
    c[:, 0] = [-0.35, 0.35]
    paths = gt_proposals(reference(), c)
    assert np.array_equal(paths[:, -1], np.repeat(reference()[None, -1], 2, axis=0))
    assert np.all(np.diff(np.concatenate([np.zeros((2, 1)), paths[:, :, 0]], axis=1), axis=1) > 0)
    assert not np.array_equal(paths[0], paths[1])


def test_high_score_does_not_override_reverse_motion():
    path = -reference()
    policy = TargetPolicy(min_signed_speed_mps=-0.02)
    physical, masks = physical_mask(path[None], reference(), policy, [3, 0], [0, 0])
    assert not physical[0] and not masks["backward_distance"][0]
    assert score_mask(np.ones((1, 7)), policy)[0]


def test_direction_and_partial_safety_labels_rejected():
    labels = np.ones((3, 7))
    labels[0, 5] = 0.5
    labels[1, 0] = 0.5
    labels[2, -1] = 0.9499999999
    assert not score_mask(labels, TargetPolicy()).any()


def test_selection_contains_all_qualified_without_duplication():
    paths = np.zeros((4000, 8, 3))
    labels = np.ones((4000, 7))
    eligible = np.zeros(4000, bool)
    eligible[3500:3800] = True
    ids, hi = select_bank(paths, labels, eligible)
    assert len(ids) == 3072 and len(set(ids)) == 3072
    assert eligible[ids[hi]].all()
    assert len(hi) == 256


def folders(tmp_path):
    root = tmp_path / "source"
    stage = tmp_path / "stage"
    target = root / "scenes/train/token"
    target.mkdir(parents=True)
    (target / "RESULT.json").write_text(json.dumps({"original": True}))
    (target / "candidates.npy").write_bytes(b"original labels and paths")
    stage.mkdir()
    (stage / "RESULT.json").write_text(json.dumps({"recovery_id": "run"}))
    (stage / "candidates.npy").write_bytes(b"new verified bank")
    cfg = dict(source=str(root), run_id="run")
    return target, stage, cfg


def test_publish_preserves_original_bytes(tmp_path):
    target, stage, cfg = folders(tmp_path)
    backup = Path(publish_stage(stage, "token", cfg))
    assert (backup / "candidates.npy").read_bytes() == b"original labels and paths"
    assert (target / "candidates.npy").read_bytes() == b"new verified bank"
    assert not stage.exists()
    assert publish_stage(stage, "token", cfg) == str(backup)


def test_publish_failure_restores_original(tmp_path, monkeypatch):
    from drivesta_feedback.batch import simscale_recovery as module

    target, stage, cfg = folders(tmp_path)
    replace = module.os.replace

    def fail(source, destination):
        if Path(source) == stage:
            raise OSError("simulated publication interruption")
        return replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        publish_stage(stage, "token", cfg)
    assert (target / "candidates.npy").read_bytes() == b"original labels and paths"
    assert stage.exists()


def test_resume_between_backup_and_publish(tmp_path):
    from drivesta_feedback.batch.simscale_recovery import os

    target, stage, cfg = folders(tmp_path)
    backup = Path(cfg["source"]) / "previous_attempts/run/train/token"
    backup.parent.mkdir(parents=True)
    os.replace(target, backup)
    publish_stage(stage, "token", cfg)
    assert (backup / "candidates.npy").read_bytes() == b"original labels and paths"
    assert (target / "candidates.npy").read_bytes() == b"new verified bank"
