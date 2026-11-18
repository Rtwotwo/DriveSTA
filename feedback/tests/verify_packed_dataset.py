"""Read real packed scenes into scorer-sized CPU tensor batches."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from pathlib import Path
import numpy as np
import torch
from drivesta_feedback.batch.pack import PackedDataset
from drivesta_feedback.batch.store import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", type=Path, required=True)
    args = parser.parse_args()
    counts = []
    checks = 0
    splits = read_json(args.packed / "COMPLETE.json")["splits"]
    for split in splits:
        dataset = PackedDataset(args.packed, split)
        for i in range(len(dataset)):
            info, rows = dataset[i]
            assert info["split"] == split
            counts.append(len(rows))
            ids = np.random.default_rng(20261014 + i).choice(len(rows), 64, replace=False)
            t = torch.from_numpy(rows["trajectories"][ids].copy())
            y = torch.from_numpy(rows["labels"][ids].copy())
            assert t.shape == (64, 8, 3) and y.shape == (64, 7)
            assert torch.isfinite(t).all() and torch.isfinite(y).all()
            assert (y >= 0).all() and (y <= 1).all()
            checks += len(ids)
    report = dict(
        passed=True,
        scenes=len(counts),
        candidates=sum(counts),
        sampled_tensor_rows=checks,
        splits=splits,
        tensor_shapes=[[64, 8, 3], [64, 7]],
    )
    assert report["candidates"] == read_json(args.packed / "COMPLETE.json")["candidates"]
    write_json(args.packed / "reader_validation.json", report)
    print(report)


if __name__ == "__main__":
    main()
