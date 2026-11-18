"""Explicit server integration check using the existing single-scene dataset.

Run separately from unit tests. Scores nine candidates and renders one BEV set.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    original = root / "feedback/runs/single_scene_20261013"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(name, argv):
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, *argv], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print((output / f"{name}.log").read_text(encoding="utf-8")[-6000:], flush=True)
            raise RuntimeError(f"{name} failed: {result.returncode}")
        print(f"PASS {name}", flush=True)

    entries = sorted(
        p for p in (root / "feedback/scripts").glob("*.py") if not p.name.startswith("_")
    )
    for path in entries:
        run("help_" + path.stem, [str(path), "--help"])
    run(
        "score_nine",
        [
            "feedback/scripts/generate_single_scene.py",
            "--output",
            str(output / "nine"),
            "--mode",
            "full",
            "--rx",
            "0.1",
            "--ry",
            "0.1",
            "--spacing",
            "0.1",
            "--workers",
            "1",
        ],
    )
    before = np.load(original / "full_v1/candidates.npz")
    before_scores = np.load(original / "full_v1/official_scores.npz")
    after = np.load(output / "nine/candidates.npz")
    after_scores = np.load(output / "nine/official_scores.npz")
    original_keys = {
        tuple(np.rint(offset / 0.05).astype(int)): i for i, offset in enumerate(before["offsets"])
    }
    score_rows = {
        int(idx): row for idx, row in zip(before_scores["indices"], before_scores["scores"])
    }
    diffs = []
    trajectory_diffs = []
    bitwise_equal = True
    for index, row in zip(after_scores["indices"], after_scores["scores"]):
        old_index = original_keys[tuple(np.rint(after["offsets"][index] / 0.05).astype(int))]
        new_trajectory, old_trajectory = (
            after["trajectories"][index],
            before["trajectories"][old_index],
        )
        bitwise_equal &= np.array_equal(new_trajectory, old_trajectory)
        trajectory_diffs.append(float(np.max(np.abs(new_trajectory - old_trajectory))))
        # The 9-row probe and original 4,141-row batch may use different FP32 kernels.
        if trajectory_diffs[-1] >= 1e-7:
            raise AssertionError(f"Trajectory parity failed: {trajectory_diffs[-1]}")
        diffs.append(float(np.max(np.abs(row - score_rows[old_index]))))
    if max(diffs) >= 1e-9:
        raise AssertionError(f"Official label parity failed: {max(diffs)}")
    print("PASS trajectory_and_official_label_parity", flush=True)
    run(
        "expanded_audit",
        [
            "feedback/scripts/audit_expanded.py",
            "--base",
            str(original / "full_v1"),
            "--result",
            str(original / "expanded_v2"),
            "--output",
            str(output / "expanded_audit.json"),
        ],
    )
    run(
        "expanded_bev",
        [
            "feedback/scripts/plot_expanded_bev.py",
            "--data",
            str(original / "expanded_v2"),
            "--output",
            str(output / "figures"),
        ],
    )
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "cli_help_entries": len(entries),
        "scored_candidates": len(diffs),
        "trajectories_bitwise_equal": bool(bitwise_equal),
        "max_trajectory_absolute_difference": max(trajectory_diffs),
        "trajectory_absolute_tolerance": 1e-7,
        "max_official_component_difference": max(diffs),
        "expanded_audit": json.loads((output / "expanded_audit.json").read_text()),
        "rendered_files": sorted(p.name for p in (output / "figures").iterdir()),
        "output_directory": str(output),
        "full_generation_started": False,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k != "expanded_audit"}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
