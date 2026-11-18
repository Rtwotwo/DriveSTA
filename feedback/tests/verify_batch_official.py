"""Actual official-label crash/recovery parity on nine goals of the first pilot scene."""

import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np
from drivesta_feedback.batch.store import read_json, write_json, lock, load_blocks
from drivesta_feedback.batch.worker import SceneWriter


def generate(directory, record, protocol, crash=False):
    from drivesta_feedback.batch.source import load_scene, construct, Evaluator
    import torch

    torch.set_num_threads(1)
    _, gt, cache, meta = load_scene(record, [record["token"]], protocol)
    keys = np.array([(x, y) for x in (-100, 0, 100) for y in (-40, 0, 40)], dtype=np.int32)
    with lock(directory):
        writer = SceneWriter(directory, protocol, meta)
        writer.phase("000_test", keys, gt, Evaluator(cache), construct, 1 if crash else None)
        write_json(
            Path(directory) / "test_complete.json",
            dict(new_blocks=writer.new_blocks, reused_blocks=writer.reused_blocks),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.run / "run.json")
    protocol = contract["protocol"]
    protocol["storage"]["chunk_candidates"] = 3
    protocol["sampling"]["max_unique_goals"] = 9
    import json

    root = Path(__file__).resolve().parents[2]
    job = json.loads((root / "feedback/runs/plans/pilot_v1/jobs.jsonl").read_text().splitlines()[0])
    record = dict(job["scenes"][0], split=job["split"], log_name=job["log_name"])
    out = args.run / "recovery_validation"
    out.mkdir(exist_ok=False)
    child = mp.get_context("spawn").Process(
        target=generate, args=(out / "recovered", record, protocol, True)
    )
    child.start()
    child.join(120)
    if child.is_alive():
        child.terminate()
        child.join()
        raise RuntimeError("Crash probe timed out")
    if child.exitcode != 75:
        raise RuntimeError("Expected committed-block crash exit")
    generate(out / "recovered", record, protocol)
    generate(out / "continuous", record, protocol)
    errors = {k: 0.0 for k in ("keys", "trajectories", "labels")}
    for (_, a), (_, b) in zip(load_blocks(out / "recovered"), load_blocks(out / "continuous")):
        for key in errors:
            errors[key] = max(errors[key], float(abs(a[key] - b[key]).max()))
    if any(errors.values()):
        raise ValueError("Real scoring recovery differs from continuous run")
    from drivesta_feedback.batch.source import load_scene, Evaluator

    _, gt, cache, _ = load_scene(record, [record["token"]], protocol)
    evaluator = Evaluator(cache)
    detail_error = 0.0
    for _, a in load_blocks(out / "recovered"):
        for trajectory, label in zip(a["trajectories"], a["labels"]):
            measured, _ = evaluator.score(trajectory, details=True)
            detail_error = max(detail_error, float(abs(measured - label).max()))
    if detail_error > 1e-9:
        raise ValueError("Official entrypoint parity failed")
    report = dict(
        passed=True,
        scene=record["token"],
        candidates=9,
        forced_exit_code=child.exitcode,
        recovered=read_json(out / "recovered/test_complete.json"),
        continuous=read_json(out / "continuous/test_complete.json"),
        max_abs_difference=errors,
        official_detail_error=detail_error,
    )
    write_json(out / "summary.json", report)
    print(report)


if __name__ == "__main__":
    main()
