import json
from pathlib import Path
import numpy as np
import pytest
from drivesta_feedback.epdms.dataset import DTYPE, SCHEMA, read_scene, read_target_pairs
from drivesta_feedback.epdms.sampling import construct, unique_ranked
from drivesta_feedback.epdms.recovery_search import (
    timed_reference,
    previous_reference,
    perturb_many,
)
from drivesta_feedback.epdms.recovery_store import (
    preserve,
    install,
    resume_transactions,
    workspace,
    freeze_successor,
    make_stage,
)
from drivesta_feedback.batch.store import write_json, checksum


def scene(root, token, prev=None, start=1.0):
    d = root / "scenes" / "train" / token
    d.mkdir(parents=True, exist_ok=True)
    bank = np.zeros(3072, dtype=DTYPE)
    bank["family_id"] = np.arange(3072)
    bank["trajectories"][:, :, 0] = np.arange(1, 9)[None] * 0.5 + np.arange(3072)[:, None] * 0.002
    bank["components"] = bank["components_raw"] = 1
    bank["epdms"] = 1
    bank["ec"] = 1 if prev else np.nan
    bank["admissible"] = bank["physical"] = bank["tracking"] = bank["space"] = True
    np.save(d / "candidates.npy", bank)
    ids = unique_ranked(bank["trajectories"], bank["epdms"], bank["admissible"])
    np.savez(
        d / "high_score.npz",
        query_indices=ids,
        previous_family_id=ids if prev else np.full(len(ids), -1),
        **{
            k: bank[k][ids]
            for k in ("trajectories", "components_raw", "components", "ec", "epdms", "family_id")
        },
    )
    write_json(d / "contract.json", dict(ego_pose_global=[0, 0, 0]))
    write_json(d / "expert.json", {})
    s = dict(
        schema=SCHEMA,
        token=token,
        split="train",
        selected_count=256,
        previous_token=prev,
        start_time=start,
        files={
            n: checksum(d / n)
            for n in ("candidates.npy", "high_score.npz", "contract.json", "expert.json")
        },
    )
    write_json(d / "summary.json", s)
    return s, d


def test_zero_perturbation():
    x = np.zeros((8, 3))
    x[:, 0] = np.arange(1, 9) * 2
    np.testing.assert_array_equal(
        construct(x, np.zeros((3, 6))), np.repeat(x[None].astype("f4"), 3, axis=0)
    )


def test_grouped_constructor():
    x = np.zeros((8, 3))
    x[:, 0] = np.arange(1, 9) * 2
    c = np.ones((4, 6)) * 0.01
    np.testing.assert_array_equal(perturb_many(np.repeat(x[None], 4, axis=0), c), construct(x, c))


def test_timing_boundary():
    x = np.zeros((8, 3))
    x[:, 0] = np.arange(1, 9) * 2
    for v in [-0.25, 0, 0.25]:
        p = timed_reference(x, v)
        np.testing.assert_array_equal(p[-1], x[-1])
        assert np.diff(p[:, 0]).min() > 0
    with pytest.raises(ValueError):
        timed_reference(x, 10)


def test_context_transform():
    states = np.zeros((2, 41, 11))
    states[:, :, 0] = 10 + np.arange(41) * 0.2
    result = previous_reference(states, 0.5, [11, 0, 0])
    np.testing.assert_allclose(
        result[:, :, 0], np.repeat(np.arange(1, 9)[None], 2, axis=0), atol=1e-5
    )


def test_strict_threshold_and_unique():
    paths = np.zeros((4, 8, 3))
    paths[2:, :, 0] = np.arange(2)[:, None] + 1
    selected = unique_ranked(paths, np.array([1, 1, 0.95, 0.95001]), np.ones(4, bool))
    assert selected.tolist() == [0, 3]


def test_preserve_and_readback(tmp_path):
    row, d = scene(tmp_path, "a")
    saved = preserve(tmp_path, "train", "a")
    assert checksum(d / "candidates.npy") == checksum(saved / "candidates.npy")
    read_scene(saved)
    assert preserve(tmp_path, "train", "a") == saved


def test_freeze_successor(tmp_path):
    a, ad = scene(tmp_path, "a")
    b, bd = scene(tmp_path, "b", "a", 1.5)
    before = read_target_pairs(tmp_path, "train", "b")["previous_trajectories"].copy()
    freeze_successor(tmp_path, a, b)
    # Change current predecessor bytes. The successor must retain the old version.
    x = np.load(ad / "candidates.npy")
    x["trajectories"] += 100
    np.save(ad / "candidates.npy", x)
    after = read_target_pairs(tmp_path, "train", "b")["previous_trajectories"]
    np.testing.assert_array_equal(before, after)
    context = json.loads((bd / "contract.json").read_text())["previous_context"]
    assert context.startswith("work/recovery_20261028/original/")


def test_interrupted_install(tmp_path, monkeypatch):
    import shutil, os

    row, d = scene(tmp_path, "a")
    stage = workspace(tmp_path) / "staged" / "a"
    shutil.copytree(d, stage)
    s = json.loads((stage / "summary.json").read_text())
    s["new_version"] = True
    write_json(stage / "summary.json", s)
    rename = Path.rename

    def fail(source, target, *a, **kw):
        if Path(source) == stage:
            raise RuntimeError("simulated interruption")
        return rename(source, target, *a, **kw)

    with monkeypatch.context() as m:
        m.setattr(Path, "rename", fail)
        with pytest.raises(RuntimeError):
            install(tmp_path, stage, "train", "a", "test")
    assert not d.exists()
    resume_transactions(tmp_path)
    assert read_scene(d)[0]["new_version"]
    resume_transactions(tmp_path)
    assert read_scene(d)[0]["new_version"]


def test_context_corruption(tmp_path):
    a, _ = scene(tmp_path, "a")
    b, bd = scene(tmp_path, "b", "a", 1.5)
    freeze_successor(tmp_path, a, b)
    path = (
        tmp_path
        / json.loads((bd / "contract.json").read_text())["previous_context"]
        / "candidates.npy"
    )
    path.write_bytes(b"broken")
    with pytest.raises(Exception):
        read_target_pairs(tmp_path, "train", "b")


def test_previous_slot_corruption(tmp_path):
    _, d = scene(tmp_path, "b", "a", 1.5)
    with np.load(d / "high_score.npz") as z:
        high = {k: z[k] for k in z.files}
    high["previous_family_id"][0] = 1
    np.savez(d / "high_score.npz", **high)
    info = json.loads((d / "summary.json").read_text())
    info["files"]["high_score.npz"] = checksum(d / "high_score.npz")
    write_json(d / "summary.json", info)
    with pytest.raises(ValueError, match="Previous family slot"):
        read_scene(d)


def test_interrupted_stage_copy(tmp_path, monkeypatch):
    import shutil

    _, d = scene(tmp_path, "a")
    stage = workspace(tmp_path) / "staged_context/train/a"

    def fail_copy(source, dest, *a, **kw):
        Path(dest).mkdir()
        (Path(dest) / "partial").write_text("interrupted")
        raise RuntimeError("copy stopped")

    with monkeypatch.context() as m:
        m.setattr(shutil, "copytree", fail_copy)
        with pytest.raises(RuntimeError):
            make_stage(tmp_path, d, stage)
    make_stage(tmp_path, d, stage)
    read_scene(stage)


def test_pair_generation_changed_retry(tmp_path, monkeypatch):
    import drivesta_feedback.epdms.dataset as data

    _, d = scene(tmp_path, "a")
    calls = []
    actual = data._read_target_pairs

    def changing(root, split, token):
        value = actual(root, split, token)
        calls.append(1)
        if len(calls) == 1:
            info = json.loads((d / "summary.json").read_text())
            info["new"] = True
            write_json(d / "summary.json", info)
        return value

    monkeypatch.setattr(data, "_read_target_pairs", changing)
    data.read_target_pairs(tmp_path, "train", "a")
    assert len(calls) == 2
