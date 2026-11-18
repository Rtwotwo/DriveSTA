"""Publish recovered high banks with fixed-size full tables and reversible backups."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .high_score_bank import COUNT, THRESHOLD, read_scene, save_bank
from .retained_bank import read_full_scene, publish_full_scene
from .scene_store import ROW_DTYPE, row_digest, validate_rows
from .store import atomic, checksum, digest, lock, read_json, write_json

FIELDS = ("keys", "trajectories", "labels")


def path_key(path):
    path = np.array(path, dtype="<f4", copy=True)
    path[path == 0] = 0
    return path.tobytes()


def select_full(original, source, high, count=3072):
    """Keep the audited high bank first, then span the original query order."""
    ids = np.asarray(high["query_indices"])
    if ids.shape != (COUNT,) or (ids < 0).any() or (ids >= len(source["labels"])).any():
        raise ValueError("Source high indices exceed the source table")
    for k in FIELDS:
        if not np.array_equal(source[k][ids], high[k]):
            raise ValueError("Source high bank does not match full table: " + k)
    if count < COUNT or len(original["labels"]) != count:
        raise ValueError("Expected a complete original fixed-size table")
    chosen = [(1, int(i)) for i in ids]
    seen_keys = {tuple(k) for k in high["keys"]}
    seen_paths = {path_key(t) for t in high["trajectories"]}
    if len(seen_keys) != COUNT or len(seen_paths) != COUNT:
        raise ValueError("Duplicate source high candidates")
    banks = (original, source)
    for origin, bank in enumerate(banks):
        eligible = []
        local_keys, local_paths = set(seen_keys), set(seen_paths)
        for i, (k, t) in enumerate(zip(bank["keys"], bank["trajectories"])):
            key, trajectory = tuple(k), path_key(t)
            if key not in local_keys and trajectory not in local_paths:
                eligible.append(i)
                local_keys.add(key)
                local_paths.add(trajectory)
        n = min(count - len(chosen), len(eligible))
        if n:
            positions = np.linspace(0, len(eligible) - 1, n).round().astype(int)
            for j in positions:
                i = eligible[j]
                chosen.append((origin, i))
                seen_keys.add(tuple(bank["keys"][i]))
                seen_paths.add(path_key(bank["trajectories"][i]))
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("Not enough unique scored candidates for the full table")
    rows = np.zeros(count, dtype=ROW_DTYPE)
    for j, (origin, i) in enumerate(chosen):
        for k in FIELDS:
            rows[k][j] = banks[origin][k][i]
    return rows, chosen


def scene_path(root, record):
    token, split = record["token"], record["split"]
    if split not in ("train", "val") or not token or "/" in token or "\\" in token:
        raise ValueError("Invalid scene identity")
    return Path(root) / "scenes" / split / token


def transaction(config):
    revision = config["revision"]
    if not revision or Path(revision).name != revision or revision in (".", ".."):
        raise ValueError("Invalid revision")
    return Path(config["target"]) / "replacements" / revision


def prepare_plan(config):
    target, txn = Path(config["target"]), transaction(config)
    if (txn / "plan.json").exists():
        plan = read_json(txn / "plan.json")
        if plan["config"] != config:
            raise ValueError("Merge configuration changed")
        return plan
    if not read_json(target / "BACKFILL_COMPLETE.json").get("passed"):
        raise ValueError("Finish the original full-table backfill first")
    partial = read_json(target / "PARTIAL.json")
    shortfalls = {r["token"] for r in partial["shortfalls"]}
    records = [json.loads(s) for s in (target / "scenes.jsonl").read_text().splitlines()]
    by_token = {r["token"]: r for r in records}
    recovered = {}
    for root in map(Path, config["sources"]):
        if root.resolve() == target.resolve():
            raise ValueError("Source and target must differ")
        for s in (root / "scenes.jsonl").read_text().splitlines():
            r = json.loads(s)
            p = scene_path(root, r)
            if not (p / "COMPLETE.json").exists():
                continue
            token = r["token"]
            if token in recovered or token not in shortfalls or by_token.get(token) != r:
                raise ValueError("Recovery token is repeated or differs from original manifest")
            recovered[token] = dict(record=r, source=str(p))
    if len(recovered) != config["expected_scenes"]:
        raise ValueError("Unexpected number of completed recovery scenes")
    txn.mkdir(parents=True, exist_ok=True)
    snapshot = txn / "root_before"
    snapshot.mkdir(exist_ok=True)
    for name in (
        "status.json",
        "PARTIAL.json",
        "followup.json",
        "run.json",
        "scenes.jsonl",
        "BACKFILL_COMPLETE.json",
        "backfill_status.json",
    ):
        if (target / name).exists():
            shutil.copy2(target / name, snapshot / name)
    plan = dict(
        config=config,
        scenes=[recovered[t] for t in sorted(recovered)],
        started_unix=time.time(),
        original_high_scenes=partial["completed_scenes"],
        original_shortfalls=len(shortfalls),
        total_scenes=len(records),
        manifest_digest=checksum(target / "scenes.jsonl"),
    )
    write_json(txn / "plan.json", plan)
    return plan


def validate_published(directory, revision, count):
    directory = Path(directory)
    origin = read_json(directory / "MERGE_ORIGIN.json")
    if origin["revision"] != revision:
        raise ValueError("Installed scene belongs to a different replacement")
    metadata, full = read_full_scene(directory, verify=True)
    high_metadata, high = read_scene(directory, verify=True)
    if metadata != high_metadata or len(full["labels"]) != count:
        raise ValueError("Merged full/high identity or count differs")
    for k in FIELDS:
        if not np.array_equal(full[k][high["query_indices"]], high[k]):
            raise ValueError("Merged subset mapping differs: " + k)
    if len(origin["row_sources"]) != count:
        raise ValueError("Missing row provenance")
    if (directory / "SHORTFALL.json").exists() or (directory / "ERROR.json").exists():
        raise ValueError("Completed scene retains a stale failure marker")
    return dict(
        token=metadata["token"],
        split=metadata["split"],
        full_count=count,
        high_count=COUNT,
        min_high_pdms=float(high["labels"][:, -1].min() * 100),
        below95=int((full["labels"][:, -1] < THRESHOLD).sum()),
        zero_scores=int((full["labels"][:, -1] == 0).sum()),
    )


def prepare_scene(config, item):
    txn, revision = transaction(config), config["revision"]
    original = scene_path(config["target"], item["record"])
    staging = scene_path(txn / "staged", item["record"])
    if (original / "MERGE_ORIGIN.json").exists():
        return validate_published(original, revision, config["full_count"])
    if (staging / "COMPLETE.json").exists():
        return validate_published(staging, revision, config["full_count"])
    source = Path(item["source"])
    old_contract, source_contract = (read_json(p / "contract.json") for p in (original, source))
    metadata, old = read_full_scene(original)
    source_metadata, all_new = read_full_scene(source)
    high_metadata, high = read_scene(source, verify=True)
    if metadata != source_metadata or metadata != high_metadata:
        raise ValueError("Original and recovered observation identities differ")
    for key in ("labeling",):
        if old_contract["protocol"][key] != source_contract["protocol"][key]:
            raise ValueError("Scoring protocols differ")
    unit = old_contract["protocol"]["sampling"]["integer_lattice_m"]
    if source_contract["protocol"]["sampling"]["integer_lattice_m"] != unit:
        raise ValueError("Source endpoint units differ")
    audit = read_json(source / "COMPLETE.json")
    tolerance = old_contract["protocol"]["labeling"]["float64_parity_tolerance"]
    if (
        audit.get("audited_candidates") != COUNT
        or audit.get("max_official_error", float("inf")) > tolerance
    ):
        raise ValueError("Recovered high bank lacks a successful full official audit")
    rows, origins = select_full(old, all_new, high, config["full_count"])
    contract = copy.deepcopy(old_contract)
    protocol = contract["protocol"]
    protocol["dataset_version"] = revision
    protocol["sampling"] = dict(
        method="selected_from_officially_scored_banks",
        integer_lattice_m=unit,
        max_unique_goals=len(rows),
        high_pdm_raw=THRESHOLD,
    )
    protocol["storage"] = dict(
        format="all_scored_candidates_v1_plus_high_score_256",
        candidates=COUNT,
        full_candidates=len(rows),
        retain_all_scores=True,
    )
    protocol["constructor"]["row_sources"] = "MERGE_ORIGIN.json"
    protocol["constructor"]["source_constructors"] = [
        old_contract["protocol"]["constructor"],
        source_contract["protocol"]["constructor"],
    ]
    contract["run_contract"] = digest(
        dict(
            revision=revision,
            token=metadata["token"],
            source_run=source_contract["run_contract"],
            original_run=old_contract["run_contract"],
        )
    )
    contract["publication_only"] = True
    validate_rows(rows, metadata["expert_gt"], protocol)
    # This directory belongs to the current uncommitted transaction only.
    if staging.exists():
        staging.resolve().relative_to((txn / "staged").resolve())
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_json(staging / "contract.json", contract)
    shutil.copy2(original / "expert.json", staging / "expert.json")
    atomic(staging / "candidates.npy", lambda f: np.save(f, rows, allow_pickle=False))
    write_json(
        staging / "state.json",
        dict(
            version=1,
            count=len(rows),
            pending=None,
            completed_stages=["recovered_bank_publication"],
            blocks=[dict(start=0, end=len(rows), digest=row_digest(rows))],
        ),
    )
    ids, bank = save_bank(staging, rows, metadata["expert_gt"], protocol)
    if not np.array_equal(ids, np.arange(COUNT)):
        raise ValueError("Publication changed the audited high-bank selection")
    for k in FIELDS:
        if not np.array_equal(rows[k][ids], high[k]):
            raise ValueError("Publication changed audited high data")
    origin = dict(
        revision=revision,
        original=str(original),
        source=str(source),
        original_contract=old_contract,
        source_contract=source_contract,
        source_high_audit=audit,
        source_scored_candidates=len(all_new["labels"]),
        row_sources=origins,
        origin_codes={"0": "original_backup", "1": "recovered_source"},
        source_high_query_indices=high["query_indices"].tolist(),
        original_files={
            n: checksum(original / n)
            for n in (
                "contract.json",
                "state.json",
                "candidates.npy",
                "ALL_SCORES.json",
                "SHORTFALL.json",
            )
        },
        source_files={
            n: checksum(source / n)
            for n in (
                "contract.json",
                "candidates.npy",
                "ALL_SCORES.json",
                "high_score.npz",
                "COMPLETE.json",
            )
        },
    )
    write_json(staging / "MERGE_ORIGIN.json", origin)
    publish_full_scene(staging, len(rows))
    write_json(
        staging / "COMPLETE.json",
        dict(
            token=metadata["token"],
            split=metadata["split"],
            log_name=metadata["log_name"],
            format="high_score_256",
            passed=True,
            status="complete",
            candidates=COUNT,
            missing_candidates=0,
            queries=len(rows),
            qualified_queries=int((rows["labels"][:, -1] >= THRESHOLD).sum()),
            best_pdms=float(rows["labels"][:, -1].max() * 100),
            min_selected_pdms=float(rows["labels"][ids, -1].min() * 100),
            audited_candidates=COUNT,
            max_official_error=audit["max_official_error"],
            audit_origin="unchanged_source_high_bank",
            source_queries=len(all_new["labels"]),
            completed_unix=time.time(),
            revision=revision,
            **bank,
        ),
    )
    return validate_published(staging, revision, config["full_count"])


def install_scene(config, item, after_backup=None):
    txn = transaction(config)
    destination = scene_path(config["target"], item["record"])
    staging = scene_path(txn / "staged", item["record"])
    backup = scene_path(txn / "backup", item["record"])
    if (destination / "MERGE_ORIGIN.json").exists():
        if not backup.exists():
            raise ValueError("Installed replacement has no rollback backup")
        return validate_published(destination, config["revision"], config["full_count"])
    validate_published(staging, config["revision"], config["full_count"])
    origin = read_json(staging / "MERGE_ORIGIN.json")
    for n, value in origin["source_files"].items():
        if checksum(Path(item["source"]) / n) != value:
            raise ValueError("Recovered source changed after preparation")
    original = backup if backup.exists() else destination
    for n, value in origin["original_files"].items():
        if checksum(original / n) != value:
            raise ValueError("Original scene changed after preparation")
    if destination.exists():
        if backup.exists():
            raise ValueError("Both original destination and backup exist")
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.replace(destination, backup)
    if after_backup is not None:
        after_backup()
    os.replace(staging, destination)
    return validate_published(destination, config["revision"], config["full_count"])


def finalize(config, plan):
    txn, target = transaction(config), Path(config["target"])
    if checksum(target / "scenes.jsonl") != plan["manifest_digest"]:
        raise ValueError("Dataset manifest changed during publication")
    results = [
        validate_published(
            scene_path(target, i["record"]), config["revision"], config["full_count"]
        )
        for i in plan["scenes"]
    ]
    replaced = {r["token"] for r in results}
    old = read_json(txn / "root_before/PARTIAL.json")
    shortfalls = [r for r in old["shortfalls"] if r["token"] not in replaced]
    completed = old["completed_scenes"] + len(results)
    status = dict(
        state="partial" if shortfalls else "complete",
        passed=not shortfalls,
        phase="recovered_scene_publication",
        total_scenes=plan["total_scenes"],
        processed_scenes=plan["total_scenes"],
        completed_scenes=completed,
        complete_candidates=completed * COUNT,
        shortfall_scenes=len(shortfalls),
        failed_scenes=0,
        full_score_scenes=plan["total_scenes"],
        full_score_candidates=plan["total_scenes"] * config["full_count"],
        eligible_scene_fraction=completed / plan["total_scenes"],
        workers=0,
        revision=config["revision"],
        updated_unix=time.time(),
    )
    write_json(target / "status.json", status)
    write_json(
        target / ("PARTIAL.json" if shortfalls else "COMPLETE.json"),
        dict(status, shortfalls=shortfalls),
    )
    followup = read_json(txn / "root_before/followup.json")
    followup.update(
        high_score_shortfalls=len(shortfalls),
        high_score_scenes=completed,
        revision=config["revision"],
    )
    write_json(target / "followup.json", followup)
    atomic(
        target / "remaining_shortfall_tokens.txt",
        lambda f: f.write("".join(r["token"] + "\n" for r in shortfalls).encode()),
    )
    result = dict(
        status,
        replacement_complete=True,
        replaced_scenes=len(results),
        replaced_full_candidates=len(results) * config["full_count"],
        replaced_high_candidates=len(results) * COUNT,
        results=results,
        backup=str(txn / "backup"),
        source_data_preserved=True,
    )
    write_json(txn / "COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply", "verify"))
    parser.add_argument(
        "--config", type=Path, default=Path("feedback/configs/merge_recovered139.json")
    )
    parser.add_argument("--limit", type=int, default=0, help="Prepare a small initial subset only")
    args = parser.parse_args()
    config = read_json(args.config)
    with lock(config["target"]):
        plan = prepare_plan(config)
        txn = transaction(config)
        if args.action == "prepare":
            items = plan["scenes"][: args.limit] if args.limit else plan["scenes"]
            results = []
            for item in items:
                results.append(prepare_scene(config, item))
                print(
                    json.dumps(
                        dict(prepared=len(results), total=len(items), token=item["record"]["token"])
                    ),
                    flush=True,
                )
            write_json(
                txn / "PREPARED.json",
                dict(prepared=len(results), total=len(plan["scenes"]), results=results),
            )
        elif args.action == "apply":
            # Validate the entire transaction before the first directory is replaced.
            for item in plan["scenes"]:
                current = scene_path(config["target"], item["record"])
                p = (
                    current
                    if (current / "MERGE_ORIGIN.json").exists()
                    else scene_path(txn / "staged", item["record"])
                )
                validate_published(p, config["revision"], config["full_count"])
            for n, item in enumerate(plan["scenes"], 1):
                result = install_scene(config, item)
                write_json(
                    txn / "progress.json",
                    dict(installed=n, total=len(plan["scenes"]), latest=result),
                )
                print(json.dumps(dict(installed=n, total=len(plan["scenes"]))), flush=True)
            result = finalize(config, plan)
            print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
        else:
            result = [
                validate_published(
                    scene_path(config["target"], i["record"]),
                    config["revision"],
                    config["full_count"],
                )
                for i in plan["scenes"]
            ]
            print(
                json.dumps(
                    dict(
                        verified=len(result),
                        full_candidates=len(result) * config["full_count"],
                        high_candidates=len(result) * COUNT,
                    )
                )
            )


if __name__ == "__main__":
    main()
