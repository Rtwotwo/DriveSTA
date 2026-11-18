"""Real-cache invariance and low/high label replay before publishing a pilot."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from drivesta_feedback.audit.simscale_recovery import load_cache
from drivesta_feedback.batch.store import read_json, write_json


def verify(args):
    token, repairs, recovery = args
    audit = read_json(repairs / "scenes" / token / "repair.json")
    old = load_cache(audit["provenance"]["source_cache_path"])
    new = load_cache(audit["cache_path"])
    # PreparedSpace materializes this cached property without moving the car.
    # Compare identical materialized object states, not presence of a lazy field.
    old.ego_state.car_footprint.geometry
    new.ego_state.car_footprint.geometry
    for key in ("ego_state", "observation", "_alternative_observation", "_actor_tail_meta"):
        if pickle.dumps(getattr(old, key)) != pickle.dumps(getattr(new, key)):
            raise ValueError(f"{token}: changed {key}")
    if not audit["provenance"]["route_repair"]:
        for key in ("route_lane_ids", "centerline", "trajectory"):
            if pickle.dumps(getattr(old, key)) != pickle.dumps(getattr(new, key)):
                raise ValueError(f"{token}: geometry-only repair changed {key}")
    if old.drivable_area_map.tokens != new.drivable_area_map.tokens:
        raise ValueError("Map IDs changed")
    result = read_json(recovery / "queries" / token / "result.json")
    if not result["recovered"]:
        return dict(token=token, invariance=True, recovered=False)
    folder = recovery / "staged/train" / token
    contract = read_json(folder / "contract.json")
    if contract["identity"]["official_cache_path"] != audit["cache_path"]:
        raise ValueError("Training contract points at old cache")
    from drivesta_feedback.batch.high_score_bank import read_scene, collate_high_score
    from drivesta_feedback.batch.batched_official import BatchedOfficial

    item = read_scene(folder, verify=True)
    collate_high_score([item])
    full = np.load(folder / "candidates.npy", allow_pickle=False)
    indices = np.unique(
        np.r_[0, 127, 255, np.linspace(256, 3071, 13, dtype=int), np.argmin(full["labels"][:, -1])]
    )
    evaluator = BatchedOfficial(new)
    labels, _ = evaluator.score(full["trajectories"][indices])
    with np.load(folder / "construction.npz", allow_pickle=False) as z:
        error = max(
            float(abs(labels - full["labels"][indices]).max()),
            float(abs(evaluator.alternative_labels - z["alternative_labels"][indices]).max()),
        )
    expert = read_json(folder / "expert.json")
    y, _ = evaluator.score(np.asarray(expert["trajectory"], np.float32)[None])
    error = max(error, float(abs(y[0] - expert["labels"]).max()))
    if error > 1e-8:
        raise ValueError("Full-bank/expert replay mismatch")
    return dict(
        token=token, invariance=True, recovered=True, replayed=len(indices) + 1, error=error
    )


def main():
    p = argparse.ArgumentParser()
    for name in ("tokens", "repairs", "recovery", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(
            pool.map(verify, [(t, a.repairs, a.recovery) for t in a.tokens.read_text().split()])
        )
    summary = dict(
        passed=True,
        scenes=len(rows),
        recovered=sum(r["recovered"] for r in rows),
        max_error=max(r.get("error", 0) for r in rows),
        rows=rows,
    )
    write_json(a.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
