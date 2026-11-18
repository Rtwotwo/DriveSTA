"""Resumable screened-target index; original full banks are read-only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from navsim.common.release_paths import metadata_json as json
import multiprocessing as mp
from pathlib import Path
import time
import yaml
from .filter_targets import TargetPolicy, filter_scene


def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    source = json.loads(a.source_index.read_text())
    policy = TargetPolicy(**yaml.safe_load(a.policy.read_text()))
    records = source["records"][: a.limit] if a.limit else source["records"]
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        policy=asdict(policy),
        source_index=str(a.source_index.resolve()),
        cache_root=str(a.cache_root.resolve()),
        limit=a.limit,
    )
    if (a.output / "config.json").exists() and json.loads(
        (a.output / "config.json").read_text()
    ) != cfg:
        raise ValueError("Resume changes source or target policy; use a new output directory")
    atomic_json(a.output / "config.json", cfg)
    path = a.output / "scenes.jsonl"
    done = {}
    if path.exists():
        # Drop only an interrupted final JSON write; earlier malformed lines are errors.
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        valid_bytes = 0
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                with path.open("r+b") as f:
                    f.truncate(valid_bytes)
                break
            done[row["token"]] = row
            valid_bytes += len(line)
    expected = {r["token"] for r in records}
    if not set(done) <= expected:
        raise ValueError("Unexpected completed token")
    todo = [r for r in records if r["token"] not in done]
    start = time.monotonic()
    jobs = ((r, source["root"], str(a.cache_root), asdict(policy)) for r in todo)
    with (
        ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool,
        path.open("a") as stream,
    ):
        for i, result in enumerate(pool.map(filter_scene, jobs), 1):
            # Compact resumable result, preserving reasons and official recheck evidence.
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            done[result["token"]] = result
            if i % 64 == 0 or i == len(todo):
                status = dict(
                    completed=len(done),
                    total=len(records),
                    elapsed_s=time.monotonic() - start,
                    scenes_per_second=i / max(time.monotonic() - start, 1e-9),
                )
                atomic_json(a.output / "status.json", status)
                print(json.dumps(status), flush=True)
    if set(done) != expected:
        raise ValueError("Screening incomplete")
    output = dict(source)
    output["records"] = [
        dict(
            r,
            auxiliary_indices=done[r["token"]]["query_indices"],
            auxiliary_labels=done[r["token"]]["selected_labels"],
            auxiliary_status=done[r["token"]]["selection_status"],
        )
        for r in records
    ]
    output["counts"] = {s: sum(r["split"] == s for r in records) for s in ("train", "val")}
    output["target_policy"] = asdict(policy)
    output["official_cache_root"] = str(a.cache_root.resolve())
    atomic_json(a.output / "index.json", output)
    summary = dict(
        scenes=len(records),
        targets=sum(len(r["auxiliary_indices"]) for r in output["records"]),
        expert_only=sum(not r["auxiliary_indices"] for r in output["records"]),
        budget_exhausted=sum(r["verification_budget_exhausted"] for r in done.values()),
    )
    atomic_json(a.output / "COMPLETE.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
