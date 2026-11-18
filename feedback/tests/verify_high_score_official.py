"""Recheck previously scored scenes, exporting their descending top256 with PDMS>=95."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np
from drivesta_feedback.batch.store import read_json, write_json, load_blocks, atomic
from drivesta_feedback.batch.scene_store import ROW_DTYPE
from drivesta_feedback.batch.high_score_bank import select_indices, save_bank, read_scene


def verify(job):
    source, output, protocol = job
    from drivesta_feedback.batch.source import load_scene, Evaluator
    import torch

    torch.set_num_threads(1)
    directory = Path(source)
    identity = read_json(directory / "contract.json")["identity"]
    blocks = [z for _, z in load_blocks(directory)]
    rows = np.zeros(sum(len(z["keys"]) for z in blocks), dtype=ROW_DTYPE)
    start = 0
    for z in blocks:
        end = start + len(z["keys"])
        for k in ("keys", "trajectories", "labels"):
            rows[k][start:end] = z[k]
        start = end
    gt = np.asarray(identity["expert_gt"])
    ids = select_indices(rows["trajectories"], rows["labels"], gt)
    record = {k: identity[k] for k in ("token", "split", "log_name", "metric_cache_relative_path")}
    target = Path(output) / "scenes" / record["split"] / record["token"]
    write_json(
        target / "contract.json",
        dict(identity=identity, protocol=protocol, source_kind="existing_pool_replay"),
    )
    result = dict(
        **record,
        passed=len(ids) == 256,
        candidates=len(ids),
        queries=len(rows),
        missing_candidates=256 - len(ids),
        format="high_score_256",
        source_kind="existing_pool_replay",
    )
    if len(ids) < 256:
        write_json(target / "SHORTFALL.json", result)
        return result
    _, _, cache, _ = load_scene(record, [record["token"]], protocol)
    ev = Evaluator(cache)
    error = 0.0
    for i in ids:
        values, _ = ev.score(rows["trajectories"][i], details=True)
        error = max(error, float(abs(values - rows["labels"][i]).max()))
        if values[-1] < 0.95 or error > 1e-9:
            raise ValueError("Official replay disagrees")
    _, meta = save_bank(target, rows, gt, protocol)
    result.update(
        meta,
        min_pdms=float(rows["labels"][ids, -1].min() * 100),
        max_pdms=float(rows["labels"][ids, -1].max() * 100),
        max_official_error=error,
        audited_candidates=len(ids),
    )
    write_json(target / "COMPLETE.json", result)
    _, stored = read_scene(target, verify=True)
    np.testing.assert_array_equal(stored["labels"], rows["labels"][ids])
    np.testing.assert_array_equal(stored["trajectories"], rows["trajectories"][ids])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    protocol = read_json(args.reference_run / "run.json")["protocol"]
    jobs = [
        (str(s), str(args.output), protocol)
        for s in sorted((args.reference_run / "scenes").glob("*/*"))
        if (s / "COMPLETE.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(verify, jobs))
    records = [
        {k: r[k] for k in ("token", "split", "log_name", "metric_cache_relative_path")}
        for r in results
    ]
    data = "".join(json.dumps(r) + "\n" for r in records).encode()
    atomic(args.output / "scenes.jsonl", lambda f: f.write(data))
    summary = dict(
        passed=all(r["passed"] for r in results),
        scenes=len(results),
        completed_scenes=sum(r["passed"] for r in results),
        audited_trajectories=sum(r.get("audited_candidates", 0) for r in results),
        max_official_error=max(r.get("max_official_error", 0) for r in results),
        new_scene_sampling_started=False,
        full_generation_started=False,
        results=results,
    )
    write_json(args.output / ("COMPLETE.json" if summary["passed"] else "PARTIAL.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
