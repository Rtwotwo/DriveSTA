"""Real official-label replay in a separate scratch output; production is read-only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def check_one(args):
    import numpy as np
    from drivesta_feedback.batch.high_score_scene import generate
    from drivesta_feedback.batch.high_score_bank import read_scene, collate_high_score
    from drivesta_feedback.batch.retained_bank import read_full_scene
    from drivesta_feedback.batch.production import load_configuration
    from drivesta_feedback.batch.store import read_json, digest

    root, source, output, token, mode = args
    root, source, output = Path(root), Path(source), Path(output)
    old = read_json(source / "run.json")
    record = next(
        json.loads(s)
        for s in (source / "scenes.jsonl").read_text().splitlines()
        if json.loads(s)["token"] == token
    )
    original = source / "scenes" / record["split"] / token
    target = output / mode / "scenes" / record["split"] / token
    if target.exists():
        raise ValueError("Use a fresh validation output")
    if mode != "new":
        shutil.copytree(original, target)
    config, _ = load_configuration(root / "feedback/configs/high_score_256.yaml", root)
    config["_backfill_full_scores"] = mode == "backfill"
    start = time.monotonic()
    result = generate(record, [token], output / mode, old["protocol"], config, digest(old))
    meta, full = read_full_scene(target)
    batch = collate_high_score([(meta, full)])
    score = full["labels"][:, -1]
    maximum_error = None
    if mode != "shortfall":
        _, before = read_scene(original)
        _, after = read_scene(target)
        for k in ("keys", "trajectories", "query_indices"):
            np.testing.assert_array_equal(before[k], after[k])
        maximum_error = float(np.abs(before["labels"] - after["labels"]).max())
        if maximum_error > 1e-9:
            raise ValueError("Published high bank parity failed")
        for k in ("keys", "trajectories", "labels"):
            np.testing.assert_array_equal(full[k][after["query_indices"]], after[k])
        if mode == "backfill":
            for name in ("high_score.npz", "COMPLETE.json"):
                if (target / name).read_bytes() != (original / name).read_bytes():
                    raise ValueError("Backfill overwrote published file")
    else:
        rows = np.load(original / "candidates.npy", allow_pickle=False)
        for k in ("keys", "trajectories", "labels"):
            np.testing.assert_array_equal(rows[k], full[k])
    return dict(
        token=token,
        mode=mode,
        queries=len(score),
        zero_scores=int((score == 0).sum()),
        low_scores=int((score < 0.95).sum()),
        high_scores=int((score >= 0.95).sum()),
        shape=list(batch["scorer_trajectories"].shape),
        max_high_bank_label_error=maximum_error,
        elapsed_s=time.monotonic() - start,
        result=result,
        passed=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    a.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(root), str(a.source), str(a.output), token, mode)
        for token, mode in [
            ("26a52620df02526b", "backfill"),
            ("c909e11a76dd5b06", "new"),
            ("85695eb2eb2655a3", "shortfall"),
        ]
    ]
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(check_one, tasks))
    report = dict(passed=all(x["passed"] for x in results), results=results)
    (a.output / "RESULTS.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
