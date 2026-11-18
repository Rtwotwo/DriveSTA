"""Fixed-size, score-ordered training banks from officially scored trajectories."""

import json
from pathlib import Path
import numpy as np
from .scene_store import ROW_DTYPE, validate_rows
from .store import checksum, read_json, write_npz

COUNT = 256
THRESHOLD = 0.95


def select_indices(trajectories, labels, gt, count=COUNT):
    """Stable descending top-k; exact stored-path duplicates have one representative."""
    trajectories, labels = np.asarray(trajectories), np.asarray(labels)
    if trajectories.shape != (len(labels), 8, 3) or labels.shape != (len(labels), 7):
        raise ValueError("Candidate shape mismatch")
    if not np.isfinite(trajectories).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite input")
    if (labels < 0).any() or (labels > 1).any():
        raise ValueError("Expected raw official 0..1 labels")
    ids = np.flatnonzero(labels[:, -1] >= THRESHOLD)
    ids = ids[np.argsort(-labels[ids, -1], kind="stable")]

    # Numerical equality treats signed zero as equal. Storage is float32.
    def key(path):
        path = np.array(path, dtype="<f4", copy=True)
        path[path == 0] = 0
        return path.tobytes()

    seen, chosen = {key(gt)}, []
    for i in ids:
        path_key = key(trajectories[i])
        if path_key not in seen:
            seen.add(path_key)
            chosen.append(int(i))
            if len(chosen) == count:
                break
    return np.asarray(chosen, dtype=np.int64)


def validate_bank(arrays, gt, protocol):
    if len(arrays["labels"]) != COUNT:
        raise ValueError("A complete bank requires exactly 256 candidates")
    rows = np.zeros(COUNT, dtype=ROW_DTYPE)
    for field in ("keys", "trajectories", "labels"):
        rows[field] = arrays[field]
    validate_rows(rows, gt, protocol)
    if np.any(rows["labels"][:, -1] < THRESHOLD):
        raise ValueError("Every selected PDMS must be >=95")
    if np.any(np.diff(rows["labels"][:, -1]) > 0):
        raise ValueError("Bank is not sorted by descending official score")
    if len(select_indices(rows["trajectories"], rows["labels"], gt)) != COUNT:
        raise ValueError("Bank contains duplicate or unchanged GT trajectories")
    q = arrays["query_indices"]
    if (
        q.shape != (COUNT,)
        or q.dtype.kind not in "iu"
        or len(np.unique(q)) != COUNT
        or (q < 0).any()
    ):
        raise ValueError("Invalid source query indices")


def save_bank(directory, rows, gt, protocol):
    ids = select_indices(rows["trajectories"], rows["labels"], gt)
    if len(ids) != COUNT:
        raise ValueError(f"Only {len(ids)} distinct candidates meet PDMS>=95")
    arrays = {k: np.asarray(rows[k][ids]) for k in ("keys", "trajectories", "labels")}
    arrays["query_indices"] = ids
    validate_bank(arrays, gt, protocol)
    path = Path(directory) / "high_score.npz"
    write_npz(path, **arrays)
    return ids, dict(data_digest=checksum(path), data_bytes=path.stat().st_size)


def read_scene(directory, verify=False):
    directory = Path(directory)
    done = read_json(directory / "COMPLETE.json")
    if (
        not done.get("passed")
        or done["candidates"] != COUNT
        or done.get("format") != "high_score_256"
    ):
        raise ValueError("Scene has not completed its 256-candidate bank")
    path = directory / "high_score.npz"
    if verify and checksum(path) != done["data_digest"]:
        raise ValueError("Bank contents differ from completion record")
    contract = read_json(directory / "contract.json")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    validate_bank(arrays, contract["identity"]["expert_gt"], contract["protocol"])
    return contract["identity"], arrays


class HighScoreDataset:
    """Direct scene reader; partial coverage requires explicit opt-in."""

    def __init__(self, root, split="train", allow_partial=False):
        self.root = Path(root)
        if split not in ("train", "val"):
            raise ValueError("Expected train or val")
        if not allow_partial and not read_json(self.root / "COMPLETE.json").get("passed"):
            raise ValueError("Dataset is incomplete")
        records = [json.loads(s) for s in (self.root / "scenes.jsonl").read_text().splitlines()]
        records = [r for r in records if r["split"] == split]
        self.requested_scenes = len(records)
        self.records = [
            r
            for r in records
            if (self.root / "scenes" / split / r["token"] / "COMPLETE.json").exists()
        ]
        included = {r["token"] for r in self.records}
        self.missing_tokens = [r["token"] for r in records if r["token"] not in included]
        if not allow_partial and self.missing_tokens:
            raise ValueError("Complete dataset is missing scenes")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        return read_scene(self.root / "scenes" / r["split"] / r["token"])


def collate_high_score(samples):
    """CPU tensors for scorer training, with unmodified official labels alongside."""
    import torch

    if not samples:
        raise ValueError("Cannot collate an empty batch")
    metadata, arrays = zip(*samples)
    if len({m["split"] for m in metadata}) != 1:
        raise ValueError("A training batch must use a single split")
    labels = torch.from_numpy(np.stack([a["labels"] for a in arrays]))
    return dict(
        tokens=[m["token"] for m in metadata],
        log_names=[m["log_name"] for m in metadata],
        split=metadata[0]["split"],
        metadata=list(metadata),
        scorer_trajectories=torch.from_numpy(np.stack([a["trajectories"] for a in arrays])).float(),
        scorer_components=labels[..., :6].float(),
        scorer_pdm=labels[..., 6].float(),
        official_labels=labels,
    )
