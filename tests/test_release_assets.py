"""Verify public asset naming, checksum protection and immutable index aliases."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('drivesta_asset_download', ROOT / 'scripts/download_assets.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_download_plan_uses_canonical_names_without_network(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['download_assets', '--profiles', 'base', '--metadata', '--plan'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    download.main()
    rows = json.loads(capsys.readouterr().out)
    assert {r['path'] for r in rows} == {
        'weights/Checkpoints/base.ckpt', 'weights/Trajectories/Metadata/drivesta_metadata.tar.gz'}


def test_corrupt_download_never_overwrites_existing_asset(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    (root / 'docs/release').mkdir(parents=True)
    (root / 'navsim/common/compatibility').mkdir(parents=True)
    name = 'weights/Checkpoints/base.ckpt'
    (root / 'docs/release/files.json').write_text(json.dumps({'files': [
        {'path': name, 'bytes': 3, 'sha256': '0' * 64}]}))
    (root / 'navsim/common/compatibility/hub_files.json').write_text(json.dumps({name: 'previous.ckpt'}))
    corrupted = tmp_path / 'corrupt.ckpt'
    corrupted.write_bytes(b'bad')
    destination = tmp_path / 'release'
    existing = destination / name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'keep existing')
    api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha='pinned'),
                          list_repo_files=lambda *a, **kw: ['previous.ckpt'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: api, hf_hub_download=lambda *a, **kw: str(corrupted)))
    monkeypatch.setattr(download, 'ROOT', root)
    monkeypatch.setattr(sys, 'argv', ['download_assets', '--profiles', 'base', '--destination', str(destination)])
    with pytest.raises(ValueError, match='checksum mismatch'):
        download.main()
    assert existing.read_bytes() == b'keep existing'


def test_index_alias_keeps_bytes_and_rejects_conflicting_copy(tmp_path):
    from navsim.common.compatibility.indexes import INDEX_FILES, prepare_indexes
    source = tmp_path / INDEX_FILES['proposals.json']
    source.parent.mkdir(parents=True)
    raw = b'{"ordered_tokens": ["01", "02"]}\n'
    source.write_bytes(raw)
    prepare_indexes(tmp_path)
    alias = tmp_path / 'data/indexes/drivesta/proposals.json'
    assert alias.read_bytes() == raw
    prepare_indexes(tmp_path)
    alias.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Conflicting index alias'):
        prepare_indexes(tmp_path)


def test_external_index_root_resolves_alias_without_modifying_metadata(tmp_path, monkeypatch):
    from navsim.common.compatibility.indexes import INDEX_FILES, index_path
    monkeypatch.setenv('DRIVESTA_INDEX_ROOT', str(tmp_path))
    expected = tmp_path / Path(INDEX_FILES['history.json']).relative_to('data/indexes/research')
    assert index_path('/not/extracted/data/indexes/drivesta/history.json') == expected

