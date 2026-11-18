"""Release portability and archive-extraction regression tests."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from navsim.common.release_paths import metadata_json, relocate
from scripts.extract_release import extract


def test_metadata_relocation_preserves_file_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIVESTA_ROOT", str(tmp_path))
    raw = json.dumps(
        {
            "root": "/data/xinyao/navsim_workspace/DriveTraj/exp/drivetto_feedback",
            "records": [{"token": "abc", "auxiliary_labels": [[1, 0, 0.5]]}],
        }
    ).encode()
    before = hashlib.sha256(raw).hexdigest()
    decoded = metadata_json.loads(raw)
    assert decoded["root"] == str(tmp_path / "data/trajectories/navtrainval")
    assert decoded["records"] == json.loads(raw)["records"]
    assert hashlib.sha256(raw).hexdigest() == before


def test_external_roots_and_prefix_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENSCENE_DATA_ROOT", str(tmp_path / "sensors"))
    monkeypatch.setenv("NAVSIM_CACHE_ROOT", str(tmp_path / "cache"))
    assert relocate("/data/xinyao/navsim_workspace/dataset/navsim_logs/test") == str(
        tmp_path / "sensors/navsim_logs/test"
    )
    assert relocate(
        "/data/xinyao/navsim_workspace/chainflow/exp/navsim_cache/training_cache"
    ) == str(tmp_path / "cache/training_cache")
    unrelated = "/data/xinyao/navsim_workspace/dataset_backup/keep"
    assert relocate(unrelated) == unrelated


def make_archive(path, name, content=b"trajectory"):
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))


def test_extract_is_byte_exact_and_rejects_overwrite(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    make_archive(archive, "data/scene/candidates.npy")
    destination = tmp_path / "out"
    extract(archive, destination)
    assert (destination / "data/scene/candidates.npy").read_bytes() == b"trajectory"
    with pytest.raises(FileExistsError):
        extract(archive, destination)


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_extract_rejects_path_traversal(tmp_path, name):
    archive = tmp_path / "bad.tar.gz"
    make_archive(archive, name)
    with pytest.raises(ValueError):
        extract(archive, tmp_path / "out")


def test_unrecognized_metadata_keeps_its_exact_digest(tmp_path):
    import hashlib
    from navsim.common.release_paths import metadata_sha256, metadata_file_sha256

    raw = b'{"token":"one", "label": 0.95}'
    path = tmp_path / "index.json"
    path.write_bytes(raw)
    assert metadata_sha256(raw) == hashlib.sha256(raw).hexdigest()
    assert metadata_file_sha256(path) == hashlib.sha256(raw).hexdigest()
    assert metadata_sha256(raw + b" ") != metadata_sha256(raw)


def test_separate_synthetic_cache_location(monkeypatch, tmp_path):
    from navsim.common.release_paths import relocate

    monkeypatch.setenv("SIMSCALE_OFFICIAL_CACHE_ROOT", str(tmp_path / "adapted"))
    monkeypatch.setenv("SIMSCALE_SOURCE_CACHE_ROOT", str(tmp_path / "original"))
    assert relocate(
        "/data/xinyao/navsim_workspace/DriveScorer/exp/drivetto_feedback_simscale/official_cache/published/token/metric_cache.pkl"
    ) == str(tmp_path / "adapted/published/token/metric_cache.pkl")
    assert relocate(
        "/data/xinyao/navsim_workspace/dataset/navsim_cache/train_metric_synthetic/a.pkl"
    ) == str(tmp_path / "original/train_metric_synthetic/a.pkl")
