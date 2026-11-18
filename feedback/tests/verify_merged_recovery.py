"""Read every replacement through training APIs and check source rows verbatim."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from drivesta_feedback.batch.high_score_bank import HighScoreDataset, collate_high_score
from drivesta_feedback.batch.retained_bank import FullScoreDataset, read_full_scene
from drivesta_feedback.batch.merge_recovered import scene_path, transaction
from drivesta_feedback.batch.store import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("feedback/configs/merge_recovered139.json"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("feedback/runs/navtrainval_high_score256/merge139_validation.json"),
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    config = read_json(a.config)
    txn = transaction(config)
    root = Path(config["target"])
    plan = read_json(txn / "plan.json")
    result = read_json(txn / "COMPLETE.json")
    records = [i["record"] for i in plan["scenes"]]
    wanted = {r["token"] for r in records}
    checks = []
    for item in plan["scenes"]:
        record = item["record"]
        dest = scene_path(root, record)
        origin = read_json(dest / "MERGE_ORIGIN.json")
        _, full = read_full_scene(dest)
        backup = scene_path(txn / "backup", record)
        banks = [read_full_scene(backup)[1], read_full_scene(item["source"])[1]]
        mapping = np.asarray(origin["row_sources"])
        for code in (0, 1):
            mask = mapping[:, 0] == code
            for field in ("keys", "trajectories", "labels"):
                if not np.array_equal(full[field][mask], banks[code][field][mapping[mask, 1]]):
                    raise ValueError("Source values differ: " + record["token"] + " " + field)
        checks.append(
            dict(
                token=record["token"],
                split=record["split"],
                original_retained=int((mapping[:, 0] == 0).sum()),
                recovered_retained=int((mapping[:, 0] == 1).sum()),
            )
        )
    print(json.dumps(dict(source_rows_checked=len(checks) * config["full_count"])), flush=True)
    all_records = [json.loads(s) for s in (root / "scenes.jsonl").read_text().splitlines()]

    def inspect(record):
        d = scene_path(root, record)
        return record, (d / "ALL_SCORES.json").exists(), (d / "COMPLETE.json").exists()

    with ThreadPoolExecutor(max_workers=32) as pool:
        publications = list(pool.map(inspect, all_records))
    if not all(full for _, full, _ in publications):
        raise ValueError("A scene is missing its full-table publication")
    print(json.dumps(dict(manifest_scenes_scanned=len(publications))), flush=True)
    datasets = {}
    missing = []
    counters = Counter()
    batch_checks = []
    for split in ("train", "val"):
        # Construct the same published-record index as __init__, using the parallel
        # filesystem scan above. __getitem__ and collate remain the production APIs.
        full = FullScoreDataset.__new__(FullScoreDataset)
        high = HighScoreDataset.__new__(HighScoreDataset)
        for instance, kind in ((full, "full"), (high, "high")):
            instance.root = root
            instance.records = [
                r
                for r, f, h in publications
                if r["split"] == split and (f if kind == "full" else h)
            ]
            instance.requested_scenes = sum(r["split"] == split for r in all_records)
            included = {r["token"] for r in instance.records}
            instance.missing_tokens = [
                r["token"]
                for r in all_records
                if r["split"] == split and r["token"] not in included
            ]
        datasets[split] = dict(
            full_scenes=len(full), high_scenes=len(high), missing_high=len(high.missing_tokens)
        )
        missing.extend(high.missing_tokens)
        for kind, dataset, count in (("full", full, config["full_count"]), ("high", high, 256)):
            indices = [i for i, r in enumerate(dataset.records) if r["token"] in wanted]
            loader = DataLoader(
                Subset(dataset, indices), batch_size=4, num_workers=2, collate_fn=collate_high_score
            )
            seen = []
            for batch in loader:
                if tuple(batch["scorer_trajectories"].shape[1:]) != (count, 8, 3):
                    raise ValueError("Training shape mismatch")
                if tuple(batch["official_labels"].shape[1:]) != (count, 7):
                    raise ValueError("Training label shape mismatch")
                if (
                    not torch.isfinite(batch["scorer_trajectories"]).all()
                    or not torch.isfinite(batch["official_labels"]).all()
                ):
                    raise ValueError("Nonfinite training values")
                if kind == "high" and (batch["official_labels"][..., -1] < 0.95).any():
                    raise ValueError("High-score threshold failed")
                seen.extend(batch["tokens"])
            expected = {r["token"] for r in records if r["split"] == split}
            if set(seen) != expected or len(seen) != len(expected):
                raise ValueError("Training split coverage differs")
            counters[kind] += len(seen)
            batch_checks.append(
                dict(
                    split=split,
                    kind=kind,
                    scenes=len(seen),
                    candidates_per_scene=count,
                    loader_workers=2,
                )
            )
    remaining = (root / "remaining_shortfall_tokens.txt").read_text().split()
    if len(remaining) != len(set(remaining)) or set(remaining) != set(missing):
        raise ValueError("Root shortfall list differs from training reader")
    if sum(x["full_scenes"] for x in datasets.values()) != result["total_scenes"]:
        raise ValueError("Full dataset coverage differs")
    if sum(x["high_scenes"] for x in datasets.values()) != result["completed_scenes"]:
        raise ValueError("High dataset coverage differs")
    output = dict(
        passed=True,
        checked_unix=time.time(),
        replaced_scenes=len(records),
        source_row_checks=len(records) * config["full_count"],
        source_label_max_difference=0.0,
        datasets=datasets,
        training_batch_checks=batch_checks,
        missing_high_tokens=len(missing),
        rows_preserved=True,
        index_scan_workers=32,
        reader_validation="production __getitem__ and collate; identical index built by parallel metadata scan",
        checks=checks,
    )
    write_json(a.output, output)
    print(json.dumps({k: v for k, v in output.items() if k != "checks"}))


if __name__ == "__main__":
    main()
