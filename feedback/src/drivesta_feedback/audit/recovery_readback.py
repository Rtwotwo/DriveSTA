"""Validate staged recovery arrays and publish token-level inventories."""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect_arrays(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if not result["passed"] or result["max_official_label_error"] > 1e-9:
        raise ValueError("Candidate bank lacks the full official audit")
    with (
        np.load(directory / "scored.npz", allow_pickle=False) as source,
        np.load(directory / "training_arrays.npz", allow_pickle=False) as data,
    ):
        t, y, ids = data["trajectories"], data["labels"], data["high_indices"]
        if t.shape != (3072, 8, 3) or y.shape != (3072, 7) or ids.shape != (256,):
            raise ValueError("Invalid training array shapes")
        if not np.isfinite(t).all() or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError("Invalid values")
        if len({path.tobytes() for path in t}) != 3072:
            raise ValueError("Duplicate trajectories")
        selected = source["selected_indices"]
        np.testing.assert_array_equal(t[ids], source["trajectories"][selected])
        np.testing.assert_array_equal(y[ids], source["labels"][selected])
        if not source["generator_eligible"][selected].all() or (y[ids, -1] < 0.95).any():
            raise ValueError("High bank violates selection requirements")
        if (np.diff(y[ids, -1]) > 0).any():
            raise ValueError("High bank is not in descending score order")
        expected = y[:, 0] * y[:, 1] * (5 * y[:, 2] + 5 * y[:, 3] + 2 * y[:, 4]) / 12
        np.testing.assert_allclose(expected, y[:, -1], atol=1e-9, rtol=0)
    return dict(
        token=result["token"],
        seed=result["seed"],
        directory=str(directory),
        min_pdms=result["minimum_selected_pdms"],
        representatives_025m=result["distinct_representatives_025m"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--original-tokens", type=Path, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    known = args.original_tokens.read_text().split()
    if len(known) != len(set(known)):
        raise ValueError("Repeated original token")
    completed = {}
    for run in args.runs:
        summary = json.loads((args.root / run / "summary.json").read_text())
        if summary["errors"]:
            raise ValueError("Recovery run has processing errors")
        for result in summary["results"]:
            if result["passed"]:
                token = result["token"]
                if token not in known:
                    raise ValueError("Recovery outside the original shortfall cohort")
                completed[token] = inspect_arrays(args.root / run / "scenes" / token)
    remaining = sorted(set(known) - set(completed))
    (args.root / "recovered_staged_tokens.txt").write_text(
        "".join(t + "\n" for t in sorted(completed))
    )
    (args.root / "remaining_after_probe_tokens.txt").write_text(
        "".join(t + "\n" for t in remaining)
    )
    result = dict(
        original_shortfall_scenes=len(known),
        recovered_staged_scenes=len(completed),
        remaining_scenes=len(remaining),
        full_training_arrays=len(completed) * 3072,
        qualifying_high_trajectories=len(completed) * 256,
        all_high_officially_replayed=True,
        all_high_pass_generator_policy=True,
        published_to_live_bank=False,
        original_training_scene_count=102740,
        records=[completed[t] for t in sorted(completed)],
    )
    (args.root / "readback.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
