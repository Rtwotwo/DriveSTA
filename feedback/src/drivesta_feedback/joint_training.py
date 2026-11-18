"""Publish reproducible joint-training indices and consolidate final score-cache dependencies."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import numpy as np

from .batch.store import write_json, checksum

EXPECTED = {0: 31725, 1: 53318, 2: 44366, 3: 36222, 4: 30686}


def under(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def scene_record(job):
    root, row, accepted = job
    from navsim.planning.training.drivesta.data.proposal.mixed_dataset import diverse_indices

    TAIL_PROTOCOL = "navsim_v1_full_ttc_actor_tail_v1"
    folder = root / "scenes" / row["split"] / row["token"]
    contract_path = folder / "contract.json"
    contract = json.loads(contract_path.read_text())
    identity = contract["identity"]
    if any(identity[k] != row[k] for k in ("token", "log_name", "split", "source", "source_group")):
        raise ValueError("Scene identity differs: " + row["token"])
    if contract["protocol"]["scoring_protocol"] != TAIL_PROTOCOL:
        raise ValueError("Wrong scoring protocol")
    # Repaired caches must survive removal of work directories. Hard links preserve exact bytes.
    source = Path(identity["official_cache_path"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not under(source, root / "official_cache"):
        dest = root / "official_cache" / "published" / row["token"] / "metric_cache.pkl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = dest.with_suffix(".tmp")
            try:
                os.link(source, tmp)
            except OSError:
                shutil.copy2(source, tmp)
            os.replace(tmp, dest)
        if checksum(source) != checksum(dest):
            raise ValueError("Cache relocation mismatch")
        source = dest
    if identity["official_cache_path"] != str(source):
        identity["official_cache_path"] = str(source)
        write_json(contract_path, contract)
    result_path = folder / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if "contract.json" in result.get("files", {}):
            result["files"]["contract.json"] = checksum(contract_path)
            write_json(result_path, result)
    record = dict(
        row,
        domain="simscale",
        official_cache_path=str(source),
        timestamp_us=identity["timestamp_us"],
        ego_pose=identity["ego_pose"],
        sensor_root=str(root.parents[1] / "data" / "sensor_blobs" / row["source"]),
    )
    full_path = folder / "candidates.npy"
    done = json.loads((folder / "ALL_SCORES.json").read_text())
    if (
        not done.get("passed")
        or done["candidates"] != 3072
        or checksum(full_path) != done["data_digest"]
    ):
        raise ValueError("Invalid full bank: " + row["token"])
    if row["token"] not in accepted:
        return None, dict(
            token=row["token"], official_cache_path=str(source), full_sha256=done["data_digest"]
        )
    done_high = json.loads((folder / "COMPLETE.json").read_text())
    high_path = folder / "high_score.npz"
    if (
        not done_high.get("passed")
        or done_high["candidates"] != 256
        or checksum(high_path) != done_high["data_digest"]
    ):
        raise ValueError("Invalid high bank")
    full = np.load(full_path, mmap_mode="r", allow_pickle=False)
    with np.load(high_path, allow_pickle=False) as high:
        ids = high["query_indices"]
        y = high["labels"]
        paths = high["trajectories"]
        if (
            full.shape != (3072,)
            or ids.shape != (256,)
            or len(np.unique(ids)) != 256
            or (ids < 0).any()
            or (ids >= 3072).any()
        ):
            raise ValueError("Invalid bank dimensions/indices")
        if (
            not np.isfinite(y).all()
            or ((y < 0) | (y > 1)).any()
            or (y[:, 6] < 0.95).any()
            or (y[:, [0, 1, 3, 4, 5]] != 1).any()
        ):
            raise ValueError("High targets fail score/components")
        if not np.array_equal(full["trajectories"][ids], paths) or not np.array_equal(
            full["labels"][ids], y
        ):
            raise ValueError("High/full correspondence changed")
        selected = diverse_indices(paths, y, ids, 0.25)
        record["auxiliary_indices"] = selected.tolist()
        record["auxiliary_labels"] = full["labels"][selected].tolist()
    with open(row["log_path"], "rb") as stream:
        frames = pickle.load(stream)
    found = [i for i, f in enumerate(frames) if f["token"] == row["token"]]
    if len(found) != 1 or found[0] < 3:
        raise ValueError("Missing historical observations")
    f = frames[found[0]]
    if f["timestamp"] != identity["timestamp_us"] or f["log_name"] != row["log_name"]:
        raise ValueError("History alignment mismatch")
    for camera in ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0"):
        image = Path(record["sensor_root"]) / f["cams"][camera]["data_path"]
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(image)
    return record, dict(
        token=row["token"],
        official_cache_path=str(source),
        full_sha256=done["data_digest"],
        high_sha256=done_high["data_digest"],
        contract_sha256=checksum(contract_path),
    )


_WORKER = None


def init_worker(root, accepted):
    global _WORKER
    _WORKER = (root, accepted)


def process_row(row):
    try:
        return scene_record((_WORKER[0], row, _WORKER[1])), None
    except Exception as e:
        return None, dict(token=row["token"], error=repr(e))


def prepare(root, navsim_index, workers=24):
    root = Path(root).resolve()
    navsim_index = Path(navsim_index).resolve()
    frozen = root / "training_splits/20261022_8514"
    accepted = set((frozen / "simscale205k_train_tokens.txt").read_text().splitlines())
    excluded = set((frozen / "shortfall_tokens.txt").read_text().splitlines())
    rows = [json.loads(x) for x in (root / "scenes.jsonl").open()]
    tokens = {r["token"] for r in rows}
    if (
        len(rows) != 204831
        or len(tokens) != 204831
        or len(accepted) != 196317
        or len(excluded) != 8514
        or accepted & excluded
        or accepted | excluded != tokens
    ):
        raise ValueError("Frozen full/qualified/exclusion membership differs")
    out = root / "training_splits/joint_v1"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "simscale_qualified.jsonl"
    temp = manifest.with_suffix(".pending")
    counts = Counter()
    errors = []
    audit = []
    t = time.monotonic()
    with (
        temp.open("w") as stream,
        ProcessPoolExecutor(workers, initializer=init_worker, initargs=(root, accepted)) as pool,
    ):
        for i, (result, error) in enumerate(pool.map(process_row, rows, chunksize=16), 1):
            if error:
                errors.append(error)
            else:
                record, witness = result
                audit.append(witness)
                if record:
                    counts[record["source_group"]] += 1
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            if i % 2000 == 0:
                print(
                    json.dumps(
                        dict(
                            checked=i,
                            eligible=sum(counts.values()),
                            errors=len(errors),
                            seconds=round(time.monotonic() - t),
                        )
                    ),
                    flush=True,
                )
    write_json(out / "errors.json", errors)
    if errors or dict(counts) != EXPECTED:
        raise ValueError(f"Joint publication failed: {len(errors)} errors, counts {counts}")
    os.replace(temp, manifest)
    real = json.loads(navsim_index.read_text())
    if len(real["records"]) != 103263 or len(set(r["token"] for r in real["records"])) != 103263:
        raise ValueError("Real index size differs")
    real_tokens = {r["token"] for r in real["records"]}
    if real_tokens & tokens:
        raise ValueError("Real/synthetic token overlap")
    # Required real feature and scoring files must exist before publishing a launchable index.
    for row in real["records"]:
        for file in ("chainflow_vla_feature.gz", "chainflow_vla_target.gz"):
            p = Path(real["cache_path"]) / row["log_name"] / row["token"] / file
            if not p.is_file():
                raise FileNotFoundError(p)
        p = (
            Path(real["official_cache_root"])
            / row["log_name"]
            / "unknown"
            / row["token"]
            / "metric_cache.pkl"
        )
        if not p.is_file():
            raise FileNotFoundError(p)
    missing = set(real["excluded_tokens"]) | excluded
    if len(missing) != 8539:
        raise ValueError("Real exclusions differ")
    exclusion = out / "excluded_tokens.txt"
    exclusion.write_text("\n".join(sorted(missing)) + "\n")
    for name, groups, size in [
        ("simscale134k", [0, 1, 2], 232672),
        ("simscale205k", [0, 1, 2, 3, 4], 299580),
    ]:
        chosen = {r["token"] for r in rows if r["source_group"] in groups} & accepted
        if chosen != set((frozen / (name + "_train_tokens.txt")).read_text().splitlines()):
            raise ValueError("Source-group membership differs")
        spec = dict(
            format="joint_feedback_v1",
            dataset=name,
            data_size=size,
            source_groups=groups,
            navsim_index=str(navsim_index),
            navsim_index_sha256=checksum(navsim_index),
            navsim_official_cache_root=real["official_cache_root"],
            simscale_root=str(root),
            simscale_manifest=str(manifest),
            simscale_manifest_sha256=checksum(manifest),
            excluded_tokens=str(exclusion),
            excluded_tokens_sha256=checksum(exclusion),
            long_gt="real_only_batch_mean_mask",
            sampling="uniform_scene_epoch",
            auxiliary_min_ade_m=0.25,
            scoring="nominal_labels_dual_tail_qualification",
        )
        write_json(out / (name + ".json"), spec)
    with (out / "dependency_audit.jsonl").open("w") as stream:
        for row in audit:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_json(
        out / "COMPLETE.json",
        dict(
            passed=True,
            full_scenes=len(rows),
            qualified=sum(counts.values()),
            excluded=len(excluded),
            groups=dict(counts),
            counts={"simscale134k": 232672, "simscale205k": 299580},
            checked_raw_images=4 * len(accepted),
            seconds=time.monotonic() - t,
        ),
    )
    print("Published " + str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--navsim-index", required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    prepare(args.root, args.navsim_index, args.workers)
