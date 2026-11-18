import json
from pathlib import Path
import pytest
from drivesta_feedback.training_splits import inspect_published, partition


def test_subset_intersection_keeps_variant_suffixes():
    rows = [
        {"token": "same-000", "source_group": 0},
        {"token": "same-003", "source_group": 3},
        {"token": "other-001", "source_group": 1},
    ]
    selected, good, bad = partition(rows, {"same-003", "other-001"}, [0, 1, 2])
    assert len(selected) == 2
    assert [r["token"] for r in good] == ["same-000"]
    assert [r["token"] for r in bad] == ["other-001"]


@pytest.mark.parametrize(
    "rows,excluded",
    [
        ([{"token": "a", "source_group": 0}] * 2, set()),
        ([{"token": "a", "source_group": 0}], {"unknown"}),
    ],
)
def test_rejects_inconsistent_universe(rows, excluded):
    with pytest.raises(ValueError):
        partition(rows, excluded, [0])


def test_fail_closed_on_published_count_or_exclusion_conflict(tmp_path):
    row = {"token": "scene-000", "split": "train"}
    folder = tmp_path / "scenes/train/scene-000"
    folder.mkdir(parents=True)
    for n in ("candidates.npy", "high_score.npz", "contract.json"):
        (folder / n).write_text("nonempty")
    (folder / "ALL_SCORES.json").write_text(
        json.dumps({"passed": True, "token": row["token"], "candidates": 3072})
    )
    marker = {
        "format": "high_score_256",
        "passed": True,
        "token": row["token"],
        "candidates": 256,
        "minimum_pdms": 95.0,
    }
    (folder / "COMPLETE.json").write_text(json.dumps(marker))
    assert inspect_published(tmp_path, row, set()) is None
    assert inspect_published(tmp_path, row, {row["token"]}) is not None
    marker["candidates"] = 255
    (folder / "COMPLETE.json").write_text(json.dumps(marker))
    assert inspect_published(tmp_path, row, set()) is not None
    (folder / "COMPLETE.json").unlink()
    assert inspect_published(tmp_path, row, {row["token"]}) is None


def test_missing_files_never_count_as_ready(tmp_path):
    assert inspect_published(tmp_path, {"token": "missing", "split": "train"}, set())
