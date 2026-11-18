import json
from pathlib import Path
import pytest
from drivesta_feedback.batch.simscale_upgrade import compatible_labels, upgrade_run


def config(digest):
    return dict(
        digest=digest,
        policy={"min_pdm": 0.95},
        actor_tail={"version": "v1"},
        scoring_protocol="full",
        full_count=3072,
        high_count=256,
        source_counts=[2],
        filter_root="filters",
        implementation={
            n: "same"
            for n in [
                "batched_official.py",
                "actor_tail.py",
                "filter_targets.py",
                "simscale_source.py",
                "physical_sampling.py",
            ]
        },
    )


def test_no_reuse_across_different_safety_threshold():
    old, new = config("old"), config("new")
    new["policy"] = {"min_pdm": 0.94}
    with pytest.raises(ValueError):
        compatible_labels(old, new)


def test_no_reuse_across_different_scorer():
    old, new = config("old"), config("new")
    new["implementation"]["batched_official.py"] = "changed"
    with pytest.raises(ValueError):
        compatible_labels(old, new)


def test_sampler_only_upgrade_allowed():
    old, new = config("old"), config("new")
    new["adaptive_sampling"] = True
    compatible_labels(old, new)


def test_upgrade_preserves_success_and_archives_incomplete(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps(dict(config=config("old"), limit=0, scenes=3)))
    for token, result in [
        ("done", dict(complete_high=True, complete_full=True)),
        ("short", dict(complete_high=False, complete_full=True)),
        ("working", None),
    ]:
        f = tmp_path / "scenes/train" / token
        f.mkdir(parents=True)
        (f / "candidates.npy").write_bytes(token.encode())
        if result is not None:
            (f / "RESULT.json").write_text(json.dumps(result))
    new = upgrade_run(tmp_path, config("new"), 0)
    assert new["compatible_digests"] == ["old"]
    assert (tmp_path / "scenes/train/done/candidates.npy").read_bytes() == b"done"
    archived = list((tmp_path / "previous_attempts").glob("*/scenes/train/*"))
    assert sorted(f.name for f in archived) == ["short", "working"]
    for f in archived:
        assert (f / "candidates.npy").read_bytes() == f.name.encode()
