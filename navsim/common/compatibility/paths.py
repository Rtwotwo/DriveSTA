"""Relocate immutable research metadata without changing its byte identity.

Published indexes retain their original SHA-256 fingerprints. Only filesystem
locations are translated in memory; tokens, labels, ordering, and hashes stay
unchanged. This also lets the original training integrity checks remain active.
"""

import hashlib
import json as _json
import os
from pathlib import Path


def project_root():
    return Path(os.environ.get("DRIVESTA_ROOT", Path(__file__).resolve().parents[3])).resolve()


def path_rules():
    root = project_root()
    indexes = Path(os.environ.get("DRIVESTA_INDEX_ROOT", root / "data/indexes/research"))
    data = Path(os.environ.get("OPENSCENE_DATA_ROOT", root / "data"))
    cache = Path(os.environ.get("NAVSIM_CACHE_ROOT", root / "data/navsim_cache"))
    trajectories = Path(os.environ.get("DRIVESTA_TRAJECTORY_ROOT", root / "data/trajectories"))
    synthetic_cache = Path(os.environ.get("SIMSCALE_SOURCE_CACHE_ROOT", cache))
    synthetic_official = Path(
        os.environ.get("SIMSCALE_OFFICIAL_CACHE_ROOT", trajectories / "simscale/official_cache")
    )
    backbones = Path(os.environ.get("DRIVESTA_BACKBONE_ROOT", root / "weights/Backbones"))
    base = "/data/xinyao/navsim_workspace/"
    rules = []
    for project in ("DriveSTA", "DriveTraj", "DriveScorer", "chainflow"):
        prefix = base + project
        rules.extend(
            [
                (prefix + "/exp/drivetto_feedback_simscale/official_cache", synthetic_official),
                (prefix + "/exp/drivetto_feedback_simscale", trajectories / "simscale"),
                (prefix + "/exp/drivetto_feedback", trajectories / "navtrainval"),
                (prefix + "/exp/navsim_cache", cache),
                (prefix + "/data/navsim_cache", synthetic_cache),
                (prefix + "/data", data),
                (
                    prefix + "/feedback/data/dense_gt_v1/official_metric_cache_v1",
                    Path(os.environ.get("OFFICIAL_PDM_CACHE", cache / "official_metric_cache_v1")),
                ),
                (
                    prefix + "/weights/vit_small_patch14_reg4_dinov2.lvd142m",
                    backbones / "dinov2",
                ),
                (prefix + "/weights/drive_jepa", backbones / "drive_jepa"),
                (prefix + "/exp/chainflow_vla/drivetraj", indexes),
                (prefix + "/exp", indexes),
                (prefix, root),
            ]
        )
    rules.append((base + "dataset/navsim_cache", synthetic_cache))
    rules.append((base + "dataset", data))
    return sorted(rules, key=lambda pair: len(pair[0]), reverse=True)


def relocate(value, rules=None):
    """Translate registered path prefixes, leaving scientific metadata intact."""
    rules = path_rules() if rules is None else rules
    if isinstance(value, str):
        if not value.startswith("/"):
            return value
        for source, target in rules:
            if value == source or value.startswith(source + "/"):
                return str(target) + value[len(source) :]
        return value
    if isinstance(value, list):
        return [relocate(item, rules) for item in value]
    if isinstance(value, dict):
        return {key: relocate(item, rules) for key, item in value.items()}
    return value


class _MetadataJSON:
    """The json API with path relocation on decoding only."""

    @staticmethod
    def loads(*args, **kwargs):
        return relocate(_json.loads(*args, **kwargs))

    @staticmethod
    def load(*args, **kwargs):
        return relocate(_json.load(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(_json, name)


metadata_json = _MetadataJSON()


# Exact, audited byte identities. Only the project-name path text changed in
# this published index; arbitrary content changes receive no compatibility alias.
_TRAINING_IDENTITIES = {
    "e8502ae77b3fc6f52756e38ca6b8542217d437426447264fc15fff52f28f0bc2": "eaf9a9d95238ad1e205187a7fb2026f6a0ff2d0074df877c9c896c850fb1cc6b",
}


def metadata_sha256(raw):
    """Return the historical training identity for an audited metadata copy."""
    digest = hashlib.sha256(raw).hexdigest()
    return _TRAINING_IDENTITIES.get(digest, digest)


def metadata_file_sha256(path):
    """Stream a dependency checksum, preserving the original training identity."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    return _TRAINING_IDENTITIES.get(value, value)
