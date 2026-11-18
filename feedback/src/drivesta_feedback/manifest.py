"""Prepare a log-separated scene manifest; no trajectory generation or scoring.

Requires PyYAML. All output files are staged and committed as one directory.
Existing output directories are never replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import tempfile

import yaml


def read_partitions(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Split configuration must be a mapping")
    membership: dict[str, str] = {}
    for split in ("train", "val", "test"):
        logs = config.get(f"{split}_logs", [])
        if not isinstance(logs, list):
            raise ValueError(f"{split}_logs must be a list")
        for log in logs:
            if not isinstance(log, str) or not log:
                raise ValueError(f"Invalid log in {split}_logs: {log!r}")
            if log in membership:
                raise ValueError(f"Repeated or overlapping log: {log}")
            membership[log] = split
    if not any(v == "train" for v in membership.values()):
        raise ValueError("No train logs configured")
    if not any(v == "val" for v in membership.values()):
        raise ValueError("No val logs configured")
    return membership


def collect_records(
    metadata: Path,
    split_config: Path,
    cache_root: Path,
    expected_train: int = 85109,
    expected_val: int = 18179,
    check_files: bool = False,
) -> tuple[list[dict], dict]:
    membership = read_partitions(split_config)
    root = cache_root.resolve()
    records: list[dict] = []
    seen: set[str] = set()
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file_name"]:
            raise ValueError(f"Expected one file_name column, got {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            source = row.get("file_name", "")
            if not source or None in row:
                raise ValueError(f"Malformed metadata row {line_number}")
            parts = PurePosixPath(source.replace("\\", "/")).parts
            if len(parts) < 4 or parts[-1] != "metric_cache.pkl":
                raise ValueError(f"Unexpected cache path at row {line_number}: {source}")
            log, scene_type, token, filename = parts[-4:]
            if any(p in ("", ".", "..", "/") for p in (log, scene_type, token)):
                raise ValueError(f"Invalid path components at row {line_number}")
            if re.fullmatch(r"[0-9a-f]{16}", token) is None:
                raise ValueError(f"Invalid scene token at row {line_number}: {token}")
            split = membership.get(log)
            if split is None:
                raise ValueError(f"Unknown log at row {line_number}: {log}")
            if split == "test":
                raise ValueError(f"Test scene in training metadata: {token}")
            if token in seen:
                raise ValueError(f"Duplicate scene token: {token}")
            seen.add(token)
            relative = PurePosixPath(log, scene_type, token, filename)
            candidate_path = root.joinpath(*relative.parts)
            if check_files and not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)
            records.append(
                {
                    "token": token,
                    "log_name": log,
                    "scene_type": scene_type,
                    "split": split,
                    "metric_cache_relative_path": relative.as_posix(),
                    "metric_cache_path": str(candidate_path),
                    "metadata_source_path": source,
                }
            )
    counts = Counter(record["split"] for record in records)
    expected = {"train": expected_train, "val": expected_val}
    for split, count in expected.items():
        if counts[split] != count:
            raise ValueError(f"{split} count {counts[split]} != expected {count}")
    records.sort(key=lambda item: (item["split"], item["log_name"], item["token"]))
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": str(metadata.resolve()),
        "split_config": str(split_config.resolve()),
        "cache_root": str(root),
        "total_scenes": len(records),
        "unique_tokens": len(seen),
        "splits": {
            split: {
                "scenes": counts[split],
                "logs": len({r["log_name"] for r in records if r["split"] == split}),
            }
            for split in ("train", "val")
        },
        "test_scenes": 0,
        "log_overlap": 0,
        "cache_file_existence": "all_checked" if check_files else "not_checked",
        "cache_contents": "not_decoded",
        "scene_timestamp_alignment": "not_checked",
        "sensor_features": "not_checked",
        "trajectory_generation_started": False,
        "training_started": False,
    }
    return records, summary


def sync_write(path: Path, lines) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for line in lines:
            stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def commit_manifest(output: Path, records: list[dict], summary: dict) -> None:
    output = output.absolute()
    if output.exists():
        raise FileExistsError(
            f"Output exists; read/reuse it or choose an explicit version: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive name reservation prevents concurrent builders of this version.
    lock_path = output.parent / f".{output.name}.manifest.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    staging = None
    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
            json.dump(
                {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "output": str(output),
                },
                lock,
            )
            lock.flush()
            os.fsync(lock.fileno())
        if output.exists():
            raise FileExistsError(output)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.pending-", dir=output.parent))
        sync_write(
            staging / "scenes.jsonl",
            (json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records),
        )
        for split in ("train", "val"):
            sync_write(
                staging / f"{split}_tokens.txt",
                (r["token"] + "\n" for r in records if r["split"] == split),
            )
        summary = dict(summary, manifest_directory_write_readback="passed")
        sync_write(
            staging / "summary.json",
            [json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"],
        )
        actual = 0
        with (staging / "scenes.jsonl").open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= len(records) or json.loads(line) != records[index]:
                    raise ValueError("Manifest readback mismatch")
                actual += 1
        if actual != len(records):
            raise ValueError("Manifest readback count mismatch")
        for split in ("train", "val"):
            actual_tokens = (
                (staging / f"{split}_tokens.txt").read_text(encoding="utf-8").splitlines()
            )
            if actual_tokens != [r["token"] for r in records if r["split"] == split]:
                raise ValueError(f"{split} token list readback mismatch")
        if json.loads((staging / "summary.json").read_text(encoding="utf-8")) != summary:
            raise ValueError("Summary readback mismatch")
        sync_directory(staging)
        staging.rename(output)
        staging = None
        sync_directory(output.parent)
    finally:
        if staging is not None and staging.exists():
            # Only the exact temporary directory created in this call is removed.
            shutil.rmtree(staging)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=85109)
    parser.add_argument("--expected-val", type=int, default=18179)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    if args.expected_train < 1 or args.expected_val < 1:
        parser.error("Both expected split counts must be positive")
    records, summary = collect_records(
        args.metadata,
        args.split_config,
        args.cache_root,
        args.expected_train,
        args.expected_val,
        args.check_files,
    )
    commit_manifest(args.output, records, summary)
    print((args.output / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
