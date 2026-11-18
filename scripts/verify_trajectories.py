"""Locate scenes, verify extracted files, and repair damaged files from a verified shard."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise ValueError(f"Unsafe inventory path: {relative}")
    target = root.joinpath(*name.parts)
    if root not in target.resolve().parents:
        raise ValueError(f"Inventory path escapes the destination: {relative}")
    return target


def load_entries(root, shard, tokens=()):
    path = safe_path(root, shard["manifest_path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != shard["manifest_sha256"]:
        raise ValueError(f"Manifest checksum mismatch: {path}")
    entries = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
    if len(entries) != shard["file_count"]:
        raise ValueError("Manifest file count mismatch")
    if tokens:
        wanted = set(tokens)
        entries = [entry for entry in entries if PurePosixPath(entry["path"]).parent.name in wanted]
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate paths in file manifest")
    return entries


def inspect_files(root, entries, quick=False):
    problems = []
    for entry in entries:
        reason = None
        try:
            path = safe_path(root, entry["path"])
            if not path.is_file():
                reason = "missing"
            elif path.stat().st_size != entry["size"]:
                reason = "size_mismatch"
            elif not quick and sha256(path) != entry["sha256"]:
                reason = "sha256_mismatch"
        except (OSError, ValueError) as error:
            reason = str(error)
        if reason:
            problems.append({"path": entry["path"], "reason": reason})
    return problems


def restore_files(root, shard, entries, problems):
    """Replace only damaged files after checking the entire archive checksum."""
    archive = safe_path(root, shard["archive_path"])
    if not archive.is_file():
        raise ValueError(f"Download this shard first: {shard['archive_path']}")
    if (
        archive.stat().st_size != shard["archive_bytes"]
        or sha256(archive) != shard["archive_sha256"]
    ):
        raise ValueError(f"Archive checksum mismatch; download again: {shard['archive_path']}")
    wanted = {item["path"] for item in problems}
    expected = {entry["path"]: entry for entry in entries if entry["path"] in wanted}
    for entry in expected.values():
        safe_path(root, entry["path"])
    restored = set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name not in expected:
                continue
            entry = expected[member.name]
            if not member.isfile() or member.size != entry["size"]:
                raise ValueError(f"Unexpected archive member: {member.name}")
            target = safe_path(root, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=target.name + ".repair-", dir=target.parent, delete=False
                ) as output:
                    temporary = Path(output.name)
                    digest = hashlib.sha256()
                    with stream.extractfile(member) as source:
                        for block in iter(lambda: source.read(1 << 20), b""):
                            output.write(block)
                            digest.update(block)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Member checksum mismatch: {member.name}")
                safe_path(root, member.name)
                temporary.chmod(0o644)
                os.replace(temporary, target)
                temporary = None
                restored.add(member.name)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            if restored == wanted:
                break
    if restored != wanted:
        raise ValueError("Required files are missing from the verified archive")
    remaining = inspect_files(root, list(expected.values()))
    if remaining:
        raise ValueError("Repaired-file readback failed: " + remaining[0]["path"])
    return len(restored)


def check_shard(root, shard, tokens=(), quick=False, repair=False):
    result = {
        "archive": shard["archive"],
        "archive_path": shard["archive_path"],
        "checked_files": 0,
        "bad_files": [],
        "repaired_files": 0,
    }
    try:
        entries = load_entries(root, shard, tokens)
        result["checked_files"] = len(entries)
        problems = inspect_files(root, entries, quick)
        result["bad_files"] = problems
        result["initial_bad_file_count"] = len(problems)
        if repair and problems:
            result["repaired_files"] = restore_files(root, shard, entries, problems)
            result["bad_files"] = []
    except (OSError, ValueError, tarfile.TarError) as error:
        result["error"] = str(error)
    return result


def find_scenes(root, index, tokens, dataset=None):
    path = safe_path(root, index["scene_catalog"]["path"])
    if sha256(path) != index["scene_catalog"]["sha256"]:
        raise ValueError("Scene-to-shard CSV checksum mismatch")
    wanted = set(tokens)
    with path.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["token"] in wanted and (dataset is None or row["dataset"] == dataset)
        ]
    found = {row["token"] for row in rows}
    if found != wanted:
        raise ValueError("Unknown scene token(s): " + ", ".join(sorted(wanted - found)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=["navtrainval", "simscale"])
    parser.add_argument(
        "--shard", action="append", default=[], help="Archive filename; repeat for several shards"
    )
    parser.add_argument(
        "--token", action="append", default=[], help="Scene token; repeat for several scenes"
    )
    parser.add_argument(
        "--locate",
        action="store_true",
        help="Print the scene-to-shard rows without reading trajectory files",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check presence and sizes only, without detecting same-size corruption",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Atomically restore only damaged files from checksum-verified local shards",
    )
    parser.add_argument("--workers", type=int, default=4, help="Shards to check concurrently")
    parser.add_argument("--report", type=Path, default=Path("trajectory_integrity_report.json"))
    args = parser.parse_args()
    if args.workers < 1 or (args.quick and args.repair) or (args.locate and not args.token):
        parser.error(
            "Use positive workers; --repair requires full SHA-256 checks; --locate requires --token"
        )
    root = args.root.resolve()
    index = json.loads((root / "manifests/index.json").read_text())
    if not index.get("complete"):
        parser.error("The catalog is incomplete")
    shards = index["shards"]
    if args.dataset:
        shards = [row for row in shards if row["dataset"] == args.dataset]
    if args.token:
        try:
            scenes = find_scenes(root, index, args.token, args.dataset)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if args.locate:
            print(json.dumps(scenes, indent=2))
            return 0
        names = {row["shard"] for row in scenes}
        shards = [row for row in shards if row["archive"] in names]
    if args.shard:
        names = {Path(name).name for name in args.shard}
        shards = [row for row in shards if row["archive"] in names]
        if {row["archive"] for row in shards} != names:
            parser.error("Unknown shard or a conflicting dataset/token filter")
    elif not args.token:
        shards = [row for row in shards if row["dataset"] != "metadata"]
    if not shards:
        parser.error("No matching shards")

    def run(shard):
        result = check_shard(root, shard, args.token, args.quick, args.repair)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "bad_files"}
                | {"remaining_bad_files": len(result["bad_files"])}
            ),
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, shards))
    failed = [row for row in results if row.get("error") or row["bad_files"]]
    report = {
        "complete": not failed,
        "mode": "sizes_only" if args.quick else "sha256",
        "root": str(root),
        "checked_shards": len(results),
        "checked_files": sum(row["checked_files"] for row in results),
        "repaired_files": sum(row["repaired_files"] for row in results),
        "bad_shards": [row["archive_path"] for row in failed],
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.report.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
