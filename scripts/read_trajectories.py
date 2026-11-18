"""Read a complete scene bank with NumPy mmap and optional high-score targets."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_scene(directory):
    """Return an immutable mmap; COMPLETE may describe only the high-score bank."""
    directory = Path(directory)
    completion = json.loads((directory / "ALL_SCORES.json").read_text())
    if not completion.get("passed"):
        raise ValueError("The full candidate bank has not passed its source checks")
    rows = np.load(directory / "candidates.npy", mmap_mode="r", allow_pickle=False)
    count = completion["candidates"]
    if count > len(rows) or not {"trajectories", "labels"}.issubset(rows.dtype.names or ()):
        raise ValueError("Invalid candidate bank schema")
    return rows[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Extracted scenes/train/<token> directory")
    args = parser.parse_args()
    rows = read_scene(args.scene)
    report = {
        "scene": str(args.scene),
        "candidates": len(rows),
        "trajectory_shape": list(rows["trajectories"].shape),
        "label_shape": list(rows["labels"].shape),
        "dtype": str(rows.dtype),
        "mean_pdms": float(rows["labels"][:, 6].mean()),
    }
    high = args.scene / "high_score.npz"
    if high.exists():
        with np.load(high, allow_pickle=False) as bank:
            report["high_score_fields"] = {key: list(bank[key].shape) for key in bank.files}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
