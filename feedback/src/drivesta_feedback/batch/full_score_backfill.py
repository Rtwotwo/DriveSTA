"""Recover full query tables after a finished first round, keeping high banks intact."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

from drivesta_feedback.paths import project_root
from .high_score_bank import read_scene as read_high_scene
from .production import load_configuration, selected_jobs, source_contract, execute
from .retained_bank import resolve_run_contract, read_full_scene
from .runner import records
from .scene_store import ROW_DTYPE
from .store import checksum, digest, lock, read_json, write_json, atomic


def require_finished_round(output, count):
    path = output / "COMPLETE.json"
    if not path.exists():
        path = output / "PARTIAL.json"
    if not path.exists():
        raise ValueError("Finish the first-round scene budget before explicit full-score recovery")
    result = read_json(path)
    if (
        result.get("state") not in ("complete", "partial")
        or result.get("total_scenes") != count
        or result.get("processed_scenes") != count
        or result.get("failed_scenes") != 0
        or result.get("completed_scenes", 0) + result.get("shortfall_scenes", 0) != count
    ):
        raise ValueError("First-round coverage is incomplete or contains generation errors")
    return result


def inspect_scene(output, record, contract_id):
    directory = output / "scenes" / record["split"] / record["token"]
    with lock(directory):
        contract = read_json(directory / "contract.json")
        if contract["run_contract"] != contract_id:
            raise ValueError("Scene belongs to another numerical run: " + record["token"])
        if any(
            contract["identity"][k] != record[k]
            for k in ("token", "split", "log_name", "metric_cache_relative_path")
        ):
            raise ValueError("Stored scene identity differs from manifest: " + record["token"])
        high_path = directory / "COMPLETE.json"
        high = read_json(high_path) if high_path.exists() else None
        if high is not None and not high.get("passed"):
            raise ValueError("Invalid high-bank completion marker: " + record["token"])
        if (directory / "ALL_SCORES.json").exists():
            _, bank = read_full_scene(directory)
            count = len(bank["labels"])
            marker = high if high is not None else read_json(directory / "SHORTFALL.json")
            if count != marker["queries"]:
                raise ValueError("Full table does not cover its recorded query budget")
            return (
                dict(
                    token=record["token"],
                    split=record["split"],
                    passed=True,
                    candidates=high["candidates"] if high else 0,
                    queries=count,
                    full_scores_available=True,
                ),
                None,
            )
        if high is None:
            raise ValueError(
                "A shortfall scene has no full table; repair separately: " + record["token"]
            )
        read_high_scene(directory, verify=True)
        return None, dict(token=record["token"], split=record["split"], queries=high["queries"])


def inventory(output, selected, contract_id, workers=32):
    complete, missing = {}, []
    start, last = time.monotonic(), 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(selected), workers * 2):
            batch = selected[offset : offset + workers * 2]
            for done, pending in pool.map(lambda r: inspect_scene(output, r, contract_id), batch):
                if done is not None:
                    complete[done["token"]] = done
                else:
                    missing.append(pending)
            scanned = offset + len(batch)
            if time.monotonic() - last >= 10 or scanned == len(selected):
                status = dict(
                    state="verified" if scanned == len(selected) else "verifying",
                    scanned_scenes=scanned,
                    total_scenes=len(selected),
                    full_score_scenes=len(complete),
                    missing_full_scenes=len(missing),
                    scan_workers=workers,
                    elapsed_s=time.monotonic() - start,
                    updated_unix=time.time(),
                )
                write_json(output / "backfill_recovery.json", status)
                print(json.dumps(dict(backfill_recovery=status)), flush=True)
                last = time.monotonic()
    summary = dict(
        total_scenes=len(selected),
        full_score_scenes=len(complete),
        full_score_candidates=sum(r["queries"] for r in complete.values()),
        missing_full_scenes=len(missing),
        missing_candidate_rows=sum(r["queries"] for r in missing),
        missing=missing,
        updated_unix=time.time(),
    )
    write_json(output / "backfill_inventory.json", summary)
    atomic(
        output / "backfill_pending_tokens.txt",
        lambda f: f.write("".join(r["token"] + "\n" for r in missing).encode()),
    )
    return complete, summary


def protected_files(output):
    return {
        name: checksum(output / name)
        for name in ("run.json", "scenes.jsonl", "status.json", "COMPLETE.json", "PARTIAL.json")
        if (output / name).exists()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "run"))
    parser.add_argument("--config", type=Path, default=Path("feedback/configs/high_score_256.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--scan-workers", type=int, default=32)
    args = parser.parse_args()
    if min(args.workers, args.scan_workers) < 1:
        parser.error("Worker counts must be positive")
    root = project_root()
    config, protocol = load_configuration(root / args.config, root)
    if config.get("objective") != "high_score_256" or not config["storage"]["retain_all_scores"]:
        parser.error("Requires high-score configuration with full-score retention enabled")
    output = root / (args.output or config["output"])
    if not (output / "run.json").exists():
        raise ValueError("Explicit recovery requires an existing dataset")
    jobs = selected_jobs(config, protocol, root, None)
    selected = records(jobs)
    contract = source_contract(root, protocol, config, selected)
    with lock(output):
        identity = resolve_run_contract(output, contract)
        protocol = identity["protocol"]
        contract_id = digest(identity)
        encoded = "".join(json.dumps(r) + "\n" for r in selected).encode()
        if (output / "scenes.jsonl").read_bytes() != encoded:
            raise ValueError("Run manifest differs")
        first_round = require_finished_round(output, len(selected))
        preserved = protected_files(output)
        if args.action == "run":
            write_json(
                output / "followup.json",
                dict(
                    requested=True,
                    action="backfill",
                    prerequisite="finished first round; explicit user-requested full-score recovery",
                    state="verifying_full_scores",
                    high_score_shortfalls=first_round["shortfall_scenes"],
                ),
            )
        completed, summary = inventory(output, selected, contract_id, args.scan_workers)
        if args.action == "inspect":
            print(json.dumps({k: v for k, v in summary.items() if k != "missing"}), flush=True)
            return
        required = summary["missing_candidate_rows"] * ROW_DTYPE.itemsize
        required += summary["missing_full_scenes"] * (
            50000 + config["estimate"]["uncached_scene_allowance_mib"] * 2**20
        )
        required += config["execution"]["minimum_free_gib"] * 2**30
        if shutil.disk_usage(output).free < required:
            raise OSError(28, "Insufficient storage for full-score recovery")
        config["_backfill_full_scores"] = True
        write_json(
            output / "backfill_execution.json",
            dict(
                started_unix=time.time(),
                workers=args.workers,
                scan_workers=args.scan_workers,
                config=config,
                source_contract=contract,
                dispatcher_digest=checksum(Path(__file__)),
                protected_files=preserved,
                initial_full_scenes=len(completed),
                missing_full_scenes=summary["missing_full_scenes"],
                missing_candidate_rows=summary["missing_candidate_rows"],
                high_score_shortfalls=first_round["shortfall_scenes"],
            ),
        )
        write_json(
            output / "followup.json",
            dict(
                requested=True,
                action="backfill",
                state="running",
                high_score_shortfalls=first_round["shortfall_scenes"],
                prerequisite="finished first round; explicit user-requested full-score recovery",
            ),
        )
        try:
            result = execute(jobs, output, protocol, config, contract_id, completed, args.workers)
            if not result["passed"] or result["full_score_scenes"] != len(selected):
                raise ValueError("Full-score coverage remains incomplete")
            if protected_files(output) != preserved:
                raise ValueError("Recovery changed protected first-round records")
        except BaseException:
            write_json(
                output / "followup.json",
                dict(
                    requested=True,
                    action="backfill",
                    state="interrupted_or_failed",
                    high_score_shortfalls=first_round["shortfall_scenes"],
                ),
            )
            raise
        write_json(
            output / "followup.json",
            dict(
                requested=True,
                action="backfill",
                state="complete",
                high_score_shortfalls=first_round["shortfall_scenes"],
                full_score_scenes=result["full_score_scenes"],
            ),
        )


if __name__ == "__main__":
    main()
