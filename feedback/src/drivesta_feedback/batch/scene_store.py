"""One mmap trajectory table per scene, used for recovery and training alike."""

import hashlib
import os
from pathlib import Path
import numpy as np
from .store import atomic, read_json, write_json, validate_block, checksum

ROW_DTYPE = np.dtype(
    [
        ("keys", "<i4", (2,)),
        ("trajectories", "<f4", (8, 3)),
        ("labels", "<f8", (7,)),
        ("phase", "<i2"),
    ],
    align=True,
)


def row_digest(rows):
    return hashlib.sha256(rows.tobytes()).hexdigest()


def validate_rows(rows, gt, protocol):
    gt = np.asarray(gt)
    validate_block(rows)
    if not len(rows):
        return
    if protocol["constructor"].get("format") == "physical_simscale_v1":
        from .simscale_dataset import validate_physical_rows

        validate_physical_rows(rows, protocol["constructor"])
        return
    if protocol["constructor"].get("format") == "verified_mixed_recovery_v1":
        from .mixed_provenance import validate_mixed_rows

        validate_mixed_rows(rows, protocol["constructor"])
        return
    goals = np.asarray(gt)[-1, :2] + rows["keys"] * protocol["sampling"]["integer_lattice_m"]
    error = np.linalg.norm(rows["trajectories"][:, -1, :2] - goals, axis=1).max()
    if error >= protocol["constructor"]["max_planned_endpoint_error_m"]:
        raise ValueError("Stored endpoint differs from requested goal")
    u = np.linspace(0, 1, 8)
    beta = 10 * u**3 - 15 * u**4 + 6 * u**5
    expected = np.asarray(gt)[None, :, :2] + beta[None, :, None] * (goals - gt[-1, :2])[:, None, :]
    if np.max(abs(expected - rows["trajectories"][:, :, :2])) > 8e-5:
        raise ValueError("Stored trajectory differs from independent quintic construction")
    y = rows["labels"]
    total = y[:, 0] * y[:, 1] * (5 * y[:, 2] + 5 * y[:, 3] + 2 * y[:, 4]) / 12
    if np.max(abs(total - y[:, 6])) > 1e-9:
        raise ValueError("Stored total differs from official PDM components")


class SceneTable:
    def __init__(self, directory, contract, capacity, allow_growth=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        identity = self.directory / "contract.json"
        if identity.exists() and read_json(identity) != contract:
            raise ValueError("Scene contract differs from this run")
        if not identity.exists():
            write_json(identity, contract)
        self.path = self.directory / "candidates.npy"
        state_path = self.directory / "state.json"
        if not self.path.exists():
            if state_path.exists():
                raise ValueError("Candidate table missing despite committed state")
            atomic(
                self.path,
                lambda f: np.save(f, np.zeros(capacity, dtype=ROW_DTYPE), allow_pickle=False),
            )
        self.data = np.load(self.path, mmap_mode="r+", allow_pickle=False)
        if allow_growth and self.data.dtype == ROW_DTYPE and self.data.ndim == 1:
            if len(self.data) < capacity:
                expanded = np.zeros(capacity, dtype=ROW_DTYPE)
                expanded[: len(self.data)] = self.data
                self.data._mmap.close()
                atomic(self.path, lambda f: np.save(f, expanded, allow_pickle=False))
                self.data = np.load(self.path, mmap_mode="r+", allow_pickle=False)
            capacity = len(self.data)  # A lower-budget invocation never truncates prior work.
        if self.data.dtype != ROW_DTYPE or self.data.shape != (capacity,):
            raise ValueError("Candidate table schema/capacity mismatch")
        self.state = (
            read_json(state_path)
            if state_path.exists()
            else dict(count=0, completed_stages=[], blocks=[], pending=None, version=1)
        )
        if not 0 <= self.state["count"] <= capacity:
            raise ValueError("Invalid committed row count")
        cursor = 0
        for block in self.state["blocks"]:
            if (
                block["start"] != cursor
                or block["end"] <= cursor
                or block["end"] > self.state["count"]
            ):
                raise ValueError("Noncontiguous committed rows")
            if row_digest(self.data[block["start"] : block["end"]]) != block["digest"]:
                raise ValueError("Committed trajectory table is corrupt")
            cursor = block["end"]
        if cursor != self.state["count"]:
            raise ValueError("Committed block count mismatch")

    @property
    def rows(self):
        return self.data[: self.state["count"]]

    def begin(self, request):
        if self.state["pending"] is not None:
            raise ValueError("Finish the pending phase before scheduling another")
        keys = np.asarray(request["keys"], dtype=np.int32).reshape(-1, 2)
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError("Duplicate requested goals")
        if {tuple(k) for k in keys} & {tuple(k) for k in self.rows["keys"]}:
            raise ValueError("Request repeats a committed goal")
        if len(keys) + len(self.rows) > len(self.data):
            raise ValueError("Request exceeds scene budget")
        self.state["pending"] = dict(request, offset=0)
        self.save_state()

    def append(self, keys, trajectories, labels, after_flush=None):
        pending = self.state["pending"]
        if pending is None or not len(keys):
            raise ValueError("No pending nonempty request")
        start, offset = self.state["count"], pending["offset"]
        end = start + len(keys)
        expected = np.asarray(pending["keys"][offset : offset + len(keys)], dtype=np.int32)
        block = np.zeros(len(keys), dtype=ROW_DTYPE)
        block["keys"], block["trajectories"], block["labels"] = keys, trajectories, labels
        block["phase"] = len(self.state["completed_stages"])
        validate_block(block, expected_keys=expected)
        self.data[start:end] = block
        self.data.flush()
        with open(self.path, "rb") as f:
            os.fsync(f.fileno())
        if row_digest(self.data[start:end]) != row_digest(block):
            raise ValueError("Candidate table readback mismatch")
        if after_flush is not None:
            after_flush()  # Crash-injection hook used by recovery tests.
        self.state["blocks"].append(dict(start=start, end=end, digest=row_digest(block)))
        self.state["count"] = end
        self.state["pending"]["offset"] += len(keys)
        self.save_state()

    def end(self):
        pending = self.state["pending"]
        if pending is None or pending["offset"] != len(pending["keys"]):
            raise ValueError("Cannot finish an incomplete phase")
        self.state["completed_stages"].append(pending["name"])
        self.state["pending"] = None
        self.save_state()

    def save_state(self):
        write_json(self.directory / "state.json", self.state)

    def close(self):
        self.data._mmap.close()


def read_scene(directory, verify=False):
    """Only a completed, audited prefix is visible to training."""
    directory = Path(directory)
    done = read_json(directory / "COMPLETE.json")
    if not done.get("passed"):
        raise ValueError("Scene has not passed completion checks")
    path = directory / "candidates.npy"
    if verify and checksum(path) != done["data_digest"]:
        raise ValueError("Completed candidate table is corrupt")
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    if rows.dtype != ROW_DTYPE or not 0 < done["candidates"] <= len(rows):
        raise ValueError("Completed table schema/count mismatch")
    return read_json(directory / "contract.json")["identity"], rows[: done["candidates"]]


class SceneDataset:
    """Lazy scene reader over the original output; no repacking or rescoring."""

    def __init__(self, root, split="train"):
        import json

        self.root = Path(root)
        if split not in ("train", "val"):
            raise ValueError("Expected an explicit train or val split")
        if not read_json(self.root / "COMPLETE.json").get("passed"):
            raise ValueError("Dataset is incomplete")
        with open(self.root / "scenes.jsonl") as f:
            self.records = [r for r in map(json.loads, f) if r["split"] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        return read_scene(self.root / "scenes" / r["split"] / r["token"])
