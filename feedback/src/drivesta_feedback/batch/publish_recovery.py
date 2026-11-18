"""Publish audited multi-reference recovery arrays in the existing scene format."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

from .store import atomic, checksum, digest, lock, read_json, write_json, write_npz
from .scene_store import ROW_DTYPE, row_digest, validate_rows
from .retained_bank import read_full_scene, publish_full_scene
from .high_score_bank import validate_bank
from .merge_recovered import scene_path, transaction, install_scene, finalize, validate_published
from .mixed_provenance import record_digest, reconstruct
from ..audit.recovery_readback import inspect_arrays


@lru_cache(maxsize=16)
def source_data(directory):
    directory = Path(directory)
    with np.load(directory / "scored.npz", allow_pickle=False) as z:
        operator = str(z["operator"]) if "operator" in z.files else "endpoint"
        if operator == "union":
            return dict(
                operator=operator,
                source_trials=z["source_trials"],
                source_trial_ids=z["source_trial_ids"],
                source_query_indices=z["source_query_indices"],
            )
        result = read_json(directory / "result.json")
        return dict(
            operator=operator,
            seed=z["seed"].tolist(),
            offset_keys=z["seed_offset_keys"].tolist(),
            record_digests={},
            seed_name=result["seed"],
            source_result=result,
        )


def source_group(directory, index):
    """Resolve a union row down to a concrete reference and perturbation."""
    group = source_data(str(directory))
    if group["operator"] == "union":
        child = Path(str(group["source_trials"][int(group["source_trial_ids"][index])]))
        return source_group(child, int(group["source_query_indices"][index]))
    return str(directory), group, int(index)


def assemble(original, old_contract, source):
    source_data.cache_clear()
    inspect_arrays(source)
    with np.load(source / "training_arrays.npz", allow_pickle=False) as z:
        paths, labels, high_ids = z["trajectories"], z["labels"], z["high_indices"]
    if not np.array_equal(high_ids, np.arange(256)):
        raise ValueError("Expected audited high bank at the front")
    with np.load(source / "scored.npz", allow_pickle=False) as z:
        selected = z["selected_indices"].copy()
    groups = [
        dict(
            operator="expert_quintic",
            seed=old_contract["identity"]["expert_gt"],
            offset_keys=original["keys"].tolist(),
            unit_m=old_contract["protocol"]["sampling"]["integer_lattice_m"],
            record_digests={},
        )
    ]
    original_lookup = {p.tobytes(): i for i, p in enumerate(original["trajectories"])}
    group_lookup = {}
    rows = np.zeros(3072, dtype=ROW_DTYPE)
    rows["trajectories"], rows["labels"] = paths, labels
    for j in range(3072):
        if j < 256:
            name, group, index = source_group(source, int(selected[j]))
            if name not in group_lookup:
                group_lookup[name] = len(groups)
                groups.append(group)
            group_id = group_lookup[name]
        else:
            index = original_lookup[paths[j].tobytes()]
            group_id = 0
            np.testing.assert_array_equal(labels[j], original["labels"][index])
        rows["keys"][j] = (group_id, index)
        rows["phase"][j] = group_id
        groups[group_id]["record_digests"][str(index)] = record_digest(paths[j], labels[j])
    # Reconstruct every new selected path before allowing publication.
    reconstruction_error = 0.0
    for group_id in range(1, len(groups)):
        mask = rows["keys"][:, 0] == group_id
        expected = reconstruct(groups[group_id], rows["keys"][mask, 1], exact=True)
        reconstruction_error = max(
            reconstruction_error, float(abs(expected - rows["trajectories"][mask]).max())
        )
        # Different vectorized batch widths may differ by a float32 ULP.
        # Publication copies the scored source bytes, never these reconstructions.
        np.testing.assert_allclose(expected, rows["trajectories"][mask], atol=1e-6, rtol=0)
    spec = dict(
        format="verified_mixed_recovery_v1",
        key_semantics="construction_group_and_query_index",
        groups=groups,
        coordinate_frame="current_ego_x_forward_y_left",
        num_poses=8,
        interval_s=0.5,
        reference="per_row_verified_reference",
        reconstruction_verified=True,
        max_reconstruction_error=reconstruction_error,
    )
    return rows, spec


def make_plan(config, readback):
    target, txn = Path(config["target"]), transaction(config)
    if (txn / "plan.json").exists():
        plan = read_json(txn / "plan.json")
        if plan["config"] != config:
            raise ValueError("Publication config changed")
        return plan
    manifest = [json.loads(s) for s in (target / "scenes.jsonl").read_text().splitlines()]
    by_token = {r["token"]: r for r in manifest}
    partial = read_json(target / "PARTIAL.json")
    missing = {r["token"] for r in partial["shortfalls"]}
    items = []
    for row in readback["records"]:
        token = row["token"]
        if token not in missing:
            raise ValueError("Only excluded shortfalls can be replaced")
        source = Path(row["directory"])
        if read_json(source / "result.json")["record"] != by_token[token]:
            raise ValueError("Recovery and production scene identities differ")
        items.append(dict(record=by_token[token], source=str(source)))
    if len(items) != config["expected_scenes"] or len({x["record"]["token"] for x in items}) != len(
        items
    ):
        raise ValueError("Unexpected recovery cohort")
    snapshot = txn / "root_before"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in (
        "status.json",
        "PARTIAL.json",
        "followup.json",
        "run.json",
        "scenes.jsonl",
        "BACKFILL_COMPLETE.json",
        "backfill_status.json",
        "remaining_shortfall_tokens.txt",
    ):
        if (target / name).exists():
            shutil.copy2(target / name, snapshot / name)
    plan = dict(
        config=config,
        scenes=items,
        started_unix=time.time(),
        original_high_scenes=partial["completed_scenes"],
        original_shortfalls=len(missing),
        total_scenes=len(manifest),
        manifest_digest=checksum(target / "scenes.jsonl"),
    )
    write_json(txn / "plan.json", plan)
    return plan


def prepare(config, item):
    txn = transaction(config)
    destination = scene_path(config["target"], item["record"])
    staging = scene_path(txn / "staged", item["record"])
    if (destination / "MERGE_ORIGIN.json").exists():
        return validate_published(destination, config["revision"], 3072)
    if (staging / "COMPLETE.json").exists():
        return validate_published(staging, config["revision"], 3072)
    source = Path(item["source"])
    source_files = {
        n: checksum(source / n) for n in ("result.json", "scored.npz", "training_arrays.npz")
    }
    old_contract = read_json(destination / "contract.json")
    metadata, old = read_full_scene(destination, verify=True)
    rows, spec = assemble(old, old_contract, source)
    contract = copy.deepcopy(old_contract)
    contract["protocol"]["constructor"] = spec
    contract["protocol"]["dataset_version"] = config["revision"]
    contract["protocol"]["storage"] = dict(
        format="all_scored_candidates_v1_plus_high_score_256",
        candidates=256,
        full_candidates=3072,
        retain_all_scores=True,
    )
    contract["protocol"]["sampling"] = dict(
        method="verified_source_rows", key_semantics=spec["key_semantics"], high_pdm_raw=0.95
    )
    contract["publication_only"] = True
    contract["run_contract"] = digest(
        dict(revision=config["revision"], identity=metadata, source_files=source_files)
    )
    validate_rows(rows, metadata["expert_gt"], contract["protocol"])
    staging.mkdir(parents=True, exist_ok=True)
    write_json(staging / "contract.json", contract)
    shutil.copy2(destination / "expert.json", staging / "expert.json")
    atomic(staging / "candidates.npy", lambda f: np.save(f, rows, allow_pickle=False))
    write_json(
        staging / "state.json",
        dict(
            version=1,
            count=3072,
            pending=None,
            completed_stages=["verified_recovery_publication"],
            blocks=[dict(start=0, end=3072, digest=row_digest(rows))],
        ),
    )
    # Preserve the safety-screened subset. A higher-score unsafe full-bank row
    # must not displace it merely because it has a larger total PDM label.
    ids = np.arange(256, dtype=np.int64)
    high = {k: rows[k][ids].copy() for k in ("keys", "trajectories", "labels")}
    high["query_indices"] = ids
    validate_bank(high, metadata["expert_gt"], contract["protocol"])
    write_npz(staging / "high_score.npz", **high)
    audit = read_json(source / "result.json")
    write_json(
        staging / "MERGE_ORIGIN.json",
        dict(
            revision=config["revision"],
            original=str(destination),
            source=str(source),
            source_files=source_files,
            original_files={
                n: checksum(destination / n)
                for n in (
                    "contract.json",
                    "state.json",
                    "candidates.npy",
                    "ALL_SCORES.json",
                    "SHORTFALL.json",
                    "expert.json",
                )
            },
            original_contract=old_contract,
            source_high_audit=audit,
            row_sources=rows["keys"].tolist(),
            construction_records="contract.json:protocol.constructor.groups",
            selection="audited_driver_eligible_subset_descending_pdm",
        ),
    )
    publish_full_scene(staging, 3072)
    write_json(
        staging / "COMPLETE.json",
        dict(
            token=metadata["token"],
            split=metadata["split"],
            log_name=metadata["log_name"],
            format="high_score_256",
            passed=True,
            status="complete",
            candidates=256,
            missing_candidates=0,
            queries=3072,
            qualified_queries=int((rows["labels"][:, -1] >= 0.95).sum()),
            best_pdms=float(rows["labels"][:, -1].max() * 100),
            min_selected_pdms=float(high["labels"][:, -1].min() * 100),
            audited_candidates=256,
            max_official_error=audit["max_official_label_error"],
            audit_origin="unchanged_fully_replayed_safety_screened_recovery",
            completed_unix=time.time(),
            revision=config["revision"],
            data_digest=checksum(staging / "high_score.npz"),
            data_bytes=(staging / "high_score.npz").stat().st_size,
        ),
    )
    return validate_published(staging, config["revision"], 3072)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply", "verify"))
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--revision", default="recovery523_20261019_v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args()
    config = dict(
        target=str(a.target.resolve()),
        readback=str(a.readback.resolve()),
        revision=a.revision,
        expected_scenes=523,
        full_count=3072,
    )
    with lock(a.target):
        plan = make_plan(config, read_json(a.readback))
        txn = transaction(config)
        if a.action == "prepare":
            items = plan["scenes"][: a.limit] if a.limit else plan["scenes"]
            with ProcessPoolExecutor(
                max_workers=a.workers, mp_context=mp.get_context("spawn")
            ) as pool:
                futures = [pool.submit(prepare, config, item) for item in items]
                for n, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if n % 16 == 0 or n == len(items):
                        print(json.dumps(dict(prepared=n, total=len(items))), flush=True)
            write_json(txn / "PREPARED.json", dict(prepared=len(items), total=len(plan["scenes"])))
        elif a.action == "apply":

            def preflight(item):
                destination = scene_path(a.target, item["record"])
                path = (
                    destination
                    if (destination / "MERGE_ORIGIN.json").exists()
                    else scene_path(txn / "staged", item["record"])
                )
                return validate_published(path, a.revision, 3072)

            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                list(pool.map(preflight, plan["scenes"]))
                futures = [pool.submit(install_scene, config, item) for item in plan["scenes"]]
                for n, future in enumerate(as_completed(futures), 1):
                    future.result()
                    write_json(txn / "progress.json", dict(installed=n, total=len(plan["scenes"])))
                    if n % 32 == 0 or n == len(plan["scenes"]):
                        print(json.dumps(dict(installed=n, total=len(plan["scenes"]))), flush=True)
            result = finalize(config, plan)
            print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
        else:
            rows = [
                validate_published(scene_path(a.target, x["record"]), a.revision, 3072)
                for x in plan["scenes"]
            ]
            write_json(txn / "READBACK.json", dict(passed=True, scenes=len(rows), results=rows))
            print(json.dumps(dict(verified=len(rows))), flush=True)


if __name__ == "__main__":
    main()
