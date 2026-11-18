"""Build deterministic log-grouped batch jobs; never generate trajectories."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import socket
import tempfile

import yaml

from drivesta_feedback.manifest import sync_write, sync_directory
from drivesta_feedback.paths import project_root


def build_jobs(records, profile, max_scenes_per_job=64):
    if max_scenes_per_job < 1:
        raise ValueError("max_scenes_per_job must be positive")
    allowed = profile["splits"]
    if not allowed or set(allowed) - {"train", "val"}:
        raise ValueError("Profiles only accept explicit train/val splits")
    tokens = set()
    log_splits = {}
    groups = defaultdict(list)
    for record in records:
        token, log, split = record["token"], record["log_name"], record["split"]
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported split: {split}")
        if token in tokens:
            raise ValueError(f"Duplicate token: {token}")
        tokens.add(token)
        if log in log_splits and log_splits[log] != split:
            raise ValueError(f"Log crosses splits: {log}")
        log_splits[log] = split
        if split in allowed:
            groups[(split, log)].append(record)
    for group in groups.values():
        group.sort(key=lambda r: r["token"])
    count = profile.get("max_scenes")
    if count is not None:
        if count < 1 or count > sum(len(g) for g in groups.values()):
            raise ValueError("Requested subset size is outside the available scene range")
        rng = random.Random(profile["selection_seed"])
        keys = sorted(groups)
        rng.shuffle(keys)
        # Prefer different logs before taking additional scenes from a log.
        first, rest = [], []
        for key in keys:
            group = list(groups[key])
            rng.shuffle(group)
            first.append(group[0])
            rest.extend(group[1:])
        rng.shuffle(rest)
        selected = (first + rest)[:count]
        groups = defaultdict(list)
        for record in selected:
            groups[(record["split"], record["log_name"])].append(record)
    jobs = []
    for (split, log), group in sorted(groups.items()):
        group.sort(key=lambda r: r["token"])
        for offset in range(0, len(group), max_scenes_per_job):
            subset = group[offset : offset + max_scenes_per_job]
            jobs.append(
                {
                    "job_id": f"{split}/{log}/{offset // max_scenes_per_job:04d}",
                    "split": split,
                    "log_name": log,
                    "scenes": [
                        {
                            "token": r["token"],
                            "scene_type": r["scene_type"],
                            "metric_cache_relative_path": r["metric_cache_relative_path"],
                        }
                        for r in subset
                    ],
                }
            )
    return jobs


def read_manifest(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "scenes.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != summary["total_scenes"]:
        raise ValueError("Manifest count differs from committed summary")
    counts = Counter(r["split"] for r in records)
    for split in ("train", "val"):
        if counts[split] != summary["splits"][split]["scenes"]:
            raise ValueError(f"{split} count mismatch")
    return records, summary


def commit_plan(output, jobs, summary, config, protocol):
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    lock_path = output.parent / f".{output.name}.plan.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    staging = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists():
            raise FileExistsError(output)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.pending-", dir=output.parent))
        sync_write(staging / "jobs.jsonl", (json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
        for name, data in (
            ("summary", summary),
            ("batch_config", config),
            ("dataset_protocol", protocol),
        ):
            sync_write(
                staging / f"{name}.json", [json.dumps(data, ensure_ascii=False, indent=2) + "\n"]
            )
        with (staging / "jobs.jsonl").open(encoding="utf-8") as stream:
            restored = [json.loads(line) for line in stream]
        if restored != jobs:
            raise ValueError("Batch plan readback mismatch")
        sync_directory(staging)
        staging.rename(output)
        staging = None
        sync_directory(output.parent)
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=["pilot", "trial", "full"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve() if args.project_root else project_root()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["mode"] != "plan_only":
        raise ValueError("This entry point supports plan_only")
    manifest_dir = root / config["manifest_directory"]
    protocol = yaml.safe_load((root / config["dataset_protocol"]).read_text(encoding="utf-8"))
    records, manifest_summary = read_manifest(manifest_dir)
    if len(records) != protocol["scene_scope"]["expected_total"]:
        raise ValueError("Dataset protocol and source manifest counts differ")
    profile = config["profiles"][args.profile]
    jobs = build_jobs(records, profile, config["scheduler"]["max_scenes_per_job"])
    counts = Counter()
    for job in jobs:
        counts[job["split"]] += len(job["scenes"])
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "source_manifest": str(manifest_dir),
        "source_scenes": len(records),
        "selected_scenes": sum(counts.values()),
        "selected_splits": dict(counts),
        "jobs": len(jobs),
        "logs": len({j["log_name"] for j in jobs}),
        "scene_workers": profile["scene_workers"],
        "max_queued_jobs": profile["scene_workers"]
        * config["scheduler"]["max_queued_jobs_per_worker"],
        "selection": "seeded_log_spread_then_remaining; labels_not_used",
        "generation_started": False,
        "training_started": False,
        "cache_contents_validated": False,
        "executor_implemented": False,
        "dataset_root": str(root / config["dataset_root"]),
    }
    commit_plan(args.output, jobs, summary, config, protocol)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
