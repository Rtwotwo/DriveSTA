"""Verify real missing/corrupt-file detection, selective repair and path confinement."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from scripts.verify_trajectories import check_shard, find_scenes


def fixture_bank(root):
    records = []
    payloads = {}
    archive_name = "drivesta_navtrainval_001-of-016.tar.gz"
    archive = root / "weights/Trajectories/Navtrainval" / archive_name
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as stream:
        for token in ("scene_a", "scene_b"):
            for filename in ("candidates.npy", "expert.json"):
                path = f"data/trajectories/navtrainval/scenes/train/{token}/{filename}"
                raw = (token + filename).encode()
                member = tarfile.TarInfo(path)
                member.size = len(raw)
                stream.addfile(member, io.BytesIO(raw))
                records.append(
                    {"path": path, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                )
                payloads[path] = raw
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
    manifest = root / "manifests/shards/files.jsonl.gz"
    manifest.parent.mkdir(parents=True)
    packed = gzip.compress("".join(json.dumps(row) + "\n" for row in records).encode())
    manifest.write_bytes(packed)
    shard = {
        "archive": archive_name,
        "archive_path": archive.relative_to(root).as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "manifest_path": manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(packed).hexdigest(),
        "dataset": "navtrainval",
        "file_count": 4,
        "scene_count": 2,
    }
    catalog = root / "manifests/scene_to_shard.csv"
    with catalog.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["dataset", "split", "token", "shard", "scene_path"]
        )
        writer.writeheader()
        for token in ("scene_a", "scene_b"):
            writer.writerow(
                {
                    "dataset": "navtrainval",
                    "split": "train",
                    "token": token,
                    "shard": archive_name,
                    "scene_path": f"data/trajectories/navtrainval/scenes/train/{token}",
                }
            )
    index = {
        "complete": True,
        "scene_catalog": {
            "path": catalog.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        },
        "shards": [shard],
    }
    (root / "manifests/index.json").write_text(json.dumps(index))
    return shard, records, payloads, index


def test_detects_same_size_corruption_and_missing_files(tmp_path):
    shard, records, _, _ = fixture_bank(tmp_path)
    (tmp_path / records[0]["path"]).write_bytes(b"x" * records[0]["size"])
    (tmp_path / records[1]["path"]).unlink()
    result = check_shard(tmp_path, shard)
    assert {row["reason"] for row in result["bad_files"]} == {"sha256_mismatch", "missing"}
    assert len(check_shard(tmp_path, shard, quick=True)["bad_files"]) == 1


def test_selective_repair_preserves_good_files_and_other_scenes(tmp_path):
    shard, records, payloads, _ = fixture_bank(tmp_path)
    (tmp_path / records[0]["path"]).write_bytes(b"bad")
    (tmp_path / records[2]["path"]).unlink()
    good = tmp_path / records[1]["path"]
    modified = good.stat().st_mtime_ns
    result = check_shard(tmp_path, shard, tokens=["scene_a"], repair=True)
    assert result["repaired_files"] == 1 and not result["bad_files"]
    assert (tmp_path / records[0]["path"]).read_bytes() == payloads[records[0]["path"]]
    assert good.stat().st_mtime_ns == modified
    assert not (tmp_path / records[2]["path"]).exists()


def test_corrupt_archive_is_rejected_before_repair(tmp_path):
    shard, records, _, _ = fixture_bank(tmp_path)
    target = tmp_path / records[0]["path"]
    target.write_bytes(b"bad")
    archive = tmp_path / shard["archive_path"]
    archive.write_bytes(archive.read_bytes() + b"corrupted")
    result = check_shard(tmp_path, shard, repair=True)
    assert "Archive checksum mismatch" in result["error"]
    assert target.read_bytes() == b"bad"


def test_manifest_corruption_is_rejected(tmp_path):
    shard, _, _, _ = fixture_bank(tmp_path)
    (tmp_path / shard["manifest_path"]).write_bytes(b"corrupt")
    assert "Manifest checksum mismatch" in check_shard(tmp_path, shard)["error"]


def test_unsafe_manifest_path_cannot_escape_root(tmp_path):
    shard, records, _, _ = fixture_bank(tmp_path)
    records[0]["path"] = "../outside.npy"
    raw = gzip.compress("".join(json.dumps(row) + "\n" for row in records).encode())
    (tmp_path / shard["manifest_path"]).write_bytes(raw)
    shard["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    result = check_shard(tmp_path, shard, repair=True)
    assert "Unsafe inventory path" in result["error"]
    assert not (tmp_path.parent / "outside.npy").exists()


def test_scene_lookup_and_cli(tmp_path):
    shard, _, _, index = fixture_bank(tmp_path)
    assert find_scenes(tmp_path, index, ["scene_a"])[0]["shard"] == shard["archive"]
    with pytest.raises(ValueError, match="Unknown scene"):
        find_scenes(tmp_path, index, ["missing"])
    script = Path(__file__).resolve().parents[1] / "scripts/verify_trajectories.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "--token", "scene_a", "--locate"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)[0]["token"] == "scene_a"
