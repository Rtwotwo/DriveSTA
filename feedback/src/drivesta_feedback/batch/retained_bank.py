"""All-query scene tables, alongside the unchanged 256-candidate training bank."""

import copy
import json
from pathlib import Path
import time
import numpy as np
from .scene_store import SceneTable, ROW_DTYPE, validate_rows
from .store import checksum, read_json, write_json


def publish_full_scene(directory, count):
    directory = Path(directory)
    contract = read_json(directory / "contract.json")
    table = SceneTable(directory, contract, count, allow_growth=True)
    try:
        if table.state["pending"] is not None or len(table.rows) != count:
            raise ValueError("Full-score publication requires a committed, finished query budget")
        validate_rows(table.rows, contract["identity"]["expert_gt"], contract["protocol"])
    finally:
        table.close()
    marker = dict(
        format="all_scored_candidates_v1",
        passed=True,
        candidates=count,
        token=contract["identity"]["token"],
        split=contract["identity"]["split"],
        data_digest=checksum(directory / "candidates.npy"),
        completed_unix=time.time(),
    )
    write_json(directory / "ALL_SCORES.json", marker)
    return marker


def read_full_scene(directory, verify=True):
    directory = Path(directory)
    marker = read_json(directory / "ALL_SCORES.json")
    if marker.get("format") != "all_scored_candidates_v1" or not marker.get("passed"):
        raise ValueError("Full-score bank is not published")
    path = directory / "candidates.npy"
    if verify and checksum(path) != marker["data_digest"]:
        raise ValueError("Full-score table contents differ from publication")
    contract = read_json(directory / "contract.json")
    state = read_json(directory / "state.json")
    n = marker["candidates"]
    if state["count"] != n or state["pending"] is not None:
        raise ValueError("Full-score bank is being extended or has incomplete state")
    if any(marker[k] != contract["identity"][k] for k in ("token", "split")):
        raise ValueError("Full-score scene identity differs")
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if rows.dtype != ROW_DTYPE or rows.ndim != 1 or not 0 < n <= len(rows):
            raise ValueError("Invalid full-score table shape")
        if verify:
            validate_rows(rows[:n], contract["identity"]["expert_gt"], contract["protocol"])
        arrays = {k: np.array(rows[k][:n], copy=True) for k in ("keys", "trajectories", "labels")}
    finally:
        rows._mmap.close()
    arrays["query_indices"] = np.arange(n, dtype=np.int64)
    return contract["identity"], arrays


def check_selected_rows(rows, bank, tolerance):
    """A replay may add missing data; it must not revise the published high bank."""
    ids = bank["query_indices"]
    if len(ids) != 256 or np.any(ids < 0) or np.any(ids >= len(rows)):
        raise ValueError("Published selected indices exceed replayed query budget")
    for key in ("keys", "trajectories"):
        if not np.array_equal(rows[key][ids], bank[key]):
            raise ValueError("Replay differs from published high-score " + key)
    if np.max(np.abs(rows["labels"][ids] - bank["labels"])) > tolerance:
        raise ValueError("Replay official labels differ from published high-score bank")


class FullScoreDataset:
    """Index only ALL_SCORES publications; never substitute a legacy top-256 bank."""

    def __init__(self, root, split="train", allow_partial=False):
        if split not in ("train", "val"):
            raise ValueError("Expected train or val")
        self.root = Path(root)
        records = [json.loads(s) for s in (self.root / "scenes.jsonl").read_text().splitlines()]
        records = [r for r in records if r["split"] == split]
        self.requested_scenes = len(records)
        self.records = [
            r
            for r in records
            if (self.root / "scenes" / split / r["token"] / "ALL_SCORES.json").exists()
        ]
        included = {r["token"] for r in self.records}
        self.missing_tokens = [r["token"] for r in records if r["token"] not in included]
        if self.missing_tokens and not allow_partial:
            raise ValueError(
                "Full-score coverage is incomplete; explicit allow_partial is required"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        return read_full_scene(self.root / "scenes" / r["split"] / r["token"])


def resolve_run_contract(output, current, upgrade=False):
    """One explicit retention-only upgrade; retain the original numerical identity."""
    output = Path(output)
    runfile, transition = output / "run.json", output / "retention_upgrade.json"
    if not runfile.exists():
        write_json(runfile, current)
        return current
    original = read_json(runfile)
    if transition.exists():
        saved = read_json(transition)
        if saved["original_contract"] != original or saved["retaining_contract"] != current:
            raise ValueError("Source/config changed after retention upgrade")
        return original
    if original == current:
        return original
    if not upgrade:
        raise ValueError("Run contract changed; retention-only upgrade needs --upgrade-retention")
    expected = copy.deepcopy(current)
    if (
        original["storage"].get("retain_all_scores") is not False
        or current["storage"].get("retain_all_scores") is not True
    ):
        raise ValueError("Only false-to-true full-score retention is supported")
    expected["storage"]["retain_all_scores"] = False
    expected["protocol"]["storage"]["retain_all_scores"] = False
    prefix = "feedback/src/drivesta_feedback/batch/"
    allowed = {prefix + name + ".py" for name in ("production", "high_score_scene")}
    added = prefix + "retained_bank.py"
    if set(current["implementation"]) != set(original["implementation"]) | {added}:
        raise ValueError("Unexpected implementation set in retention upgrade")
    expected["implementation"].pop(added)
    for name in allowed:
        expected["implementation"][name] = original["implementation"][name]
    if expected != original:
        raise ValueError("Retention upgrade changes sampling, scoring, selection or scene identity")
    write_json(
        transition,
        dict(
            version=1,
            upgraded_unix=time.time(),
            original_contract=original,
            retaining_contract=current,
            scope="Keep all subsequent scored queries; retain old scene contracts and top256 files",
        ),
    )
    return original
