"""Read completed real scene banks through the training Dataset and DataLoader."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from drivesta_feedback.batch.high_score_bank import HighScoreDataset, collate_high_score, read_scene
from drivesta_feedback.batch.store import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--scenes", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ds = HighScoreDataset(args.root, args.split, allow_partial=args.allow_partial)
    available = len(ds)
    if available < args.scenes:
        raise ValueError(f"Need {args.scenes} completed scenes, found {available}")
    ds.records = ds.records[: args.scenes]
    loader = DataLoader(
        ds,
        batch_size=min(4, args.scenes),
        num_workers=args.workers,
        collate_fn=collate_high_score,
        multiprocessing_context="spawn" if args.workers else None,
    )
    seen = []
    shapes = []
    for b in loader:
        n = len(b["tokens"])
        y = b["official_labels"]
        t = b["scorer_trajectories"]
        assert t.shape == (n, 256, 8, 3) and y.shape == (n, 256, 7)
        assert t.dtype == torch.float32 and y.dtype == torch.float64
        assert torch.isfinite(t).all() and torch.isfinite(y).all()
        assert (y[..., 6] >= 0.95).all() and (y[..., 6] <= 1).all()
        assert (torch.diff(y[..., 6], dim=1) <= 0).all()
        assert b["split"] == args.split and b["scorer_components"].shape == (n, 256, 6)
        for i, token in enumerate(b["tokens"]):
            meta, a = read_scene(args.root / "scenes" / args.split / token, verify=True)
            assert meta["token"] == token and meta["split"] == args.split
            assert meta["log_name"] == b["log_names"][i]
            np.testing.assert_array_equal(a["trajectories"], t[i].numpy())
            np.testing.assert_array_equal(a["labels"], y[i].numpy())
        seen.extend(b["tokens"])
        shapes.append(dict(trajectories=list(t.shape), labels=list(y.shape)))
    result = dict(
        passed=True,
        root=str(args.root),
        split=args.split,
        available_scenes=available,
        checked_scenes=len(seen),
        tokens=seen,
        batches=shapes,
        data_loader_workers=args.workers,
        partial_read_explicit=args.allow_partial,
        training_started=False,
    )
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
