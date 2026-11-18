"""Exercise the public concurrent CLI with independently hashed shards."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def test_parallel_extraction_and_duplicate_rejection(tmp_path):
    archives = []
    records = []
    for index in range(2):
        path = tmp_path / f"shard{index}.tar.gz"
        payload = f"scene {index}".encode()
        with tarfile.open(path, "w:gz") as stream:
            member = tarfile.TarInfo(f"data/scene{index}/candidates.npy")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
        archives.append(path)
        records.append(
            {"archive": path.name, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"archives": records}))
    destination = tmp_path / "extracted"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/extract_release.py"),
        *map(str, archives),
        "--manifest",
        str(manifest),
        "--destination",
        str(destination),
        "--workers",
        "2",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    for index in range(2):
        assert (destination / f"data/scene{index}/candidates.npy").read_text() == f"scene {index}"
    # Detect a duplicate before a worker creates files in a fresh destination.
    duplicate = tmp_path / "duplicate"
    command = [
        sys.executable,
        command[1],
        str(archives[0]),
        str(archives[0]),
        "--manifest",
        str(manifest),
        "--destination",
        str(duplicate),
        "--workers",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2 and "only once" in result.stderr
    assert not duplicate.exists()
