"""Continue a fixed shortfall cohort in a separate, directly readable dataset."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import yaml

from drivesta_feedback.paths import project_root
from .production import load_configuration, source_contract, execute, scan_completed
from .plan import build_jobs
from .retained_bank import read_full_scene
from .scene_store import row_digest
from .store import atomic, checksum, digest, lock, read_json, write_json


def distinct_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Supplement output must be separate from the source dataset")
    return source, output


def cohort(source, tokens_file, expected):
    tokens = Path(tokens_file).read_text().split()
    if len(tokens) != expected or len(set(tokens)) != expected:
        raise ValueError("Shortfall token list has duplicates or an unexpected count")
    original = read_json(source / "PARTIAL.json")
    if original["failed_scenes"] or original["processed_scenes"] != original["total_scenes"]:
        raise ValueError("Source first round is unfinished or contains errors")
    if set(tokens) != {r["token"] for r in original["shortfalls"]}:
        raise ValueError("Token list differs from the finished first-round shortfalls")
    all_records = [json.loads(s) for s in (source / "scenes.jsonl").read_text().splitlines()]
    selected = [r for r in all_records if r["token"] in set(tokens)]
    if len(selected) != expected:
        raise ValueError("Missing or repeated scene in source manifest")
    return all_records, selected


def inspect_source(source, record, contract_id):
    directory = source / "scenes" / record["split"] / record["token"]
    identity, bank = read_full_scene(directory)
    contract = read_json(directory / "contract.json")
    if contract["run_contract"] != contract_id or any(
        identity[k] != record[k]
        for k in ("token", "split", "log_name", "metric_cache_relative_path")
    ):
        raise ValueError("Source scene identity differs from the frozen run")
    if (directory / "COMPLETE.json").exists():
        raise ValueError("This recovery cohort must contain original shortfalls only")
    result = read_json(directory / "SHORTFALL.json")
    if result["queries"] != len(bank["labels"]) or result["candidates"] >= 256:
        raise ValueError("Source shortfall publication differs from its full table")
    return dict(
        record=record, result=result, expert_pdm=read_json(directory / "expert.json")["labels"][-1]
    )


def pilot_selection(items, count):
    # Include near-complete, GT-high but sampled-low, near-threshold, and difficult cases.
    groups = [[], [], [], []]
    for item in sorted(items, key=lambda x: x["record"]["token"]):
        r = item["result"]
        i = (
            0
            if r["candidates"]
            else (1 if item["expert_pdm"] >= 0.95 else (2 if r["best_pdms"] >= 90 else 3))
        )
        groups[i].append(item)
    groups[0].sort(key=lambda x: (-x["result"]["candidates"], x["record"]["token"]))
    chosen = []
    while len(chosen) < count and any(groups):
        for group in groups:
            if group and len(chosen) < count:
                chosen.append(group.pop(0))
    return chosen


def source_snapshot(directory):
    names = (
        "candidates.npy",
        "state.json",
        "contract.json",
        "expert.json",
        "ALL_SCORES.json",
        "SHORTFALL.json",
    )
    return {n: checksum(directory / n) for n in names}


def check_prefix(source_directory, destination):
    original = np.load(source_directory / "candidates.npy", mmap_mode="r", allow_pickle=False)
    copied = np.load(destination / "candidates.npy", mmap_mode="r", allow_pickle=False)
    try:
        count = read_json(source_directory / "state.json")["count"]
        if len(copied) < count or row_digest(original[:count]) != row_digest(copied[:count]):
            raise ValueError("Supplement changed the original scored prefix")
    finally:
        original._mmap.close()
        copied._mmap.close()


def prepare_scene(source, output, record, seed=None):
    original = source / "scenes" / record["split"] / record["token"]
    destination = output / "scenes" / record["split"] / record["token"]
    expected = dict(
        source=str(source),
        source_files=source_snapshot(original),
        original_queries=read_json(original / "state.json")["count"],
    )
    if destination.exists():
        if read_json(destination / "SUPPLEMENT_ORIGIN.json") != expected:
            raise ValueError("Supplement source changed since the last invocation")
        check_prefix(original, destination)
        return
    copy_from = original
    if seed is not None:
        candidate = seed / "scenes" / record["split"] / record["token"]
        if candidate.exists():
            if read_json(candidate / "SUPPLEMENT_ORIGIN.json") != expected:
                raise ValueError("Pilot belongs to a different original dataset")
            read_full_scene(candidate)
            check_prefix(original, candidate)
            copy_from = candidate
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".pending")
    try:
        # Real copies: extending a table must never mutate the source or pilot file.
        shutil.copytree(copy_from, staging, ignore=shutil.ignore_patterns(".lock"))
        write_json(staging / "SUPPLEMENT_ORIGIN.json", expected)
        read_full_scene(staging)
        check_prefix(original, staging)
        if source_snapshot(original) != expected["source_files"]:
            raise ValueError("Source changed during copy")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def summarize(output, selected, source):
    complete, short, errors, full, queries = [], [], [], 0, 0
    for record in selected:
        d = output / "scenes" / record["split"] / record["token"]
        origin = read_json(d / "SUPPLEMENT_ORIGIN.json")
        original = source / "scenes" / record["split"] / record["token"]
        if source_snapshot(original) != origin["source_files"]:
            raise ValueError("Original source files changed during supplement")
        check_prefix(original, d)
        if (d / "ALL_SCORES.json").exists():
            _, bank = read_full_scene(d)
            full += 1
            queries += len(bank["labels"])
        if (d / "COMPLETE.json").exists():
            complete.append(record["token"])
        elif (d / "ERROR.json").exists():
            errors.append(record["token"])
        else:
            short.append(read_json(d / "SHORTFALL.json"))
    result = dict(
        total_scenes=len(selected),
        completed_scenes=len(complete),
        shortfall_scenes=len(short),
        error_scenes=len(errors),
        full_score_scenes=full,
        full_score_candidates=queries,
        complete_candidates=len(complete) * 256,
        shortfalls=short,
        errors=errors,
        original_prefix_unchanged=True,
        updated_unix=time.time(),
    )
    write_json(output / "supplement_summary.json", result)
    atomic(
        output / "remaining_shortfall_tokens.txt",
        lambda f: f.write("".join(r["token"] + "\n" for r in short).encode()),
    )
    return result


def resume_budgets(output, budgets, selected):
    """Never publish an interrupted high-budget scene at a lower-stage prefix."""
    floor = budgets[0]
    phase = output / "continuation_phase.json"
    if phase.exists():
        floor = max(floor, read_json(phase)["budget"])
    for record in selected:
        d = output / "scenes" / record["split"] / record["token"]
        if not (d / "COMPLETE.json").exists():
            count = read_json(d / "state.json")["count"]
            eligible = [b for b in budgets if b >= count]
            if not eligible:
                raise ValueError("Requested budgets are below already committed queries")
            floor = max(floor, eligible[0])
    remaining = [b for b in budgets if b >= floor]
    if not remaining:
        raise ValueError("Resume cannot reduce the recorded active budget")
    return remaining


def clear_recovered_errors(output, selected):
    # Called only after execute returns without any failed scene in this cohort.
    for record in selected:
        d = output / "scenes" / record["split"] / record["token"]
        if (d / "ERROR.json").exists() and (d / "ALL_SCORES.json").exists():
            read_full_scene(d)
            if (d / "COMPLETE.json").exists() or (d / "SHORTFALL.json").exists():
                (d / "ERROR.json").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "run"))
    parser.add_argument("--config", type=Path, default=Path("feedback/configs/shortfall690.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--pilot", type=int, help="Stratified small cohort; use a separate output")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--seed-from", type=Path, help="Finished pilot directory to copy and reuse")
    args = parser.parse_args()
    if args.workers < 1 or (args.pilot is not None and not 1 <= args.pilot <= 690):
        parser.error("Invalid worker or pilot count")
    root = project_root()
    settings = yaml.safe_load((root / args.config).read_text())
    source, output = distinct_output(
        root / settings["source"], root / (args.output or settings["output"])
    )
    config, _ = load_configuration(root / settings["base_config"], root)
    all_records, selected = cohort(
        source, root / settings["tokens_file"], settings["expected_scenes"]
    )
    frozen = read_json(source / "run.json")
    transition = read_json(source / "retention_upgrade.json")
    if (
        transition["original_contract"] != frozen
        or source_contract(
            root, load_configuration(root / settings["base_config"], root)[1], config, all_records
        )
        != transition["retaining_contract"]
    ):
        raise ValueError("Current numerical implementation differs from the frozen run")
    contract_id, protocol = digest(frozen), frozen["protocol"]
    budgets = args.budgets or settings["budgets"]
    if budgets != sorted(set(budgets)) or any(
        b not in config["sampling"]["query_budgets"][1:] for b in budgets
    ):
        raise ValueError("Choose increasing, predeclared continuation budgets")
    with ThreadPoolExecutor(max_workers=16) as pool:
        items = list(pool.map(lambda r: inspect_source(source, r, contract_id), selected))
    if args.pilot:
        items = pilot_selection(items, args.pilot)
    selected = [item["record"] for item in items]
    info = dict(
        source=str(source),
        output=str(output),
        scene_count=len(selected),
        splits={s: sum(r["split"] == s for r in selected) for s in ("train", "val")},
        reused_queries=sum(i["result"]["queries"] for i in items),
        budgets=budgets,
        workers=args.workers,
        numerical_contract=contract_id,
        selection="stratified pilot" if args.pilot else "all first-round shortfalls",
    )
    print(json.dumps(info), flush=True)
    if args.action == "inspect":
        return
    if (
        shutil.disk_usage(source).free
        < config["execution"]["minimum_free_gib"] * 2**30 + len(selected) * max(budgets) * 168
    ):
        raise OSError(28, "Insufficient free space for continuation")
    seed = args.seed_from.resolve() if args.seed_from else None
    if seed is not None:
        distinct_output(seed, output)
        seed_state = read_json(seed / "status.json")["state"]
        if seed_state not in ("complete", "partial"):
            raise ValueError("Seed pilot must have finished its requested budget")
    wrapper = dict(
        version="shortfall_continuation_v1",
        original_run=contract_id,
        source=str(source),
        selected_digest=digest(selected),
        scene_count=len(selected),
        dispatcher_digest=checksum(Path(__file__)),
    )
    with lock(output):
        if (output / "run.json").exists() and read_json(output / "run.json") != wrapper:
            raise ValueError("Supplement contract changed; use its original command")
        write_json(output / "run.json", wrapper)
        atomic(
            output / "scenes.jsonl",
            lambda f: f.write("".join(json.dumps(r) + "\n" for r in selected).encode()),
        )
        write_json(output / "invocation.json", dict(info, started_unix=time.time()))
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda r: prepare_scene(source, output, r, seed), selected))
        # One scene per job keeps the 690-scene tail parallel across all workers.
        jobs = build_jobs(selected, dict(splits=["train", "val"]), max_scenes_per_job=1)
        for budget in resume_budgets(output, budgets, selected):
            completed = scan_completed(
                output, selected, contract_id=contract_id, retain_all_scores=True, scan_workers=16
            )
            if len(completed) == len(selected):
                break
            write_json(
                output / "continuation_phase.json",
                dict(
                    budget=budget,
                    completed_before=len(completed),
                    state="running",
                    updated_unix=time.time(),
                ),
            )
            stage_config = copy.deepcopy(config)
            stage_config["_query_budget"] = budget
            result = execute(
                jobs, output, protocol, stage_config, contract_id, completed, args.workers
            )
            write_json(output / f"budget_{budget}_result.json", result)
            clear_recovered_errors(output, selected)
            summarize(output, selected, source)
        result = summarize(output, selected, source)
        write_json(
            output / "continuation_phase.json",
            dict(
                budget=budgets[-1],
                state=(
                    "failed"
                    if result["error_scenes"]
                    else "partial" if result["shortfall_scenes"] else "complete"
                ),
                updated_unix=time.time(),
            ),
        )
        print(
            json.dumps({k: v for k, v in result.items() if k not in ("shortfalls", "errors")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
