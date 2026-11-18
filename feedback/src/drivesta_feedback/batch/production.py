"""128-process bounded production, per-scene commits, explicit user launch."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, FIRST_COMPLETED, wait
from collections import Counter
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import shutil
import signal
import time
import traceback
import numpy as np
import yaml
from drivesta_feedback.paths import project_root
from .plan import read_manifest, build_jobs
from .runner import records, validate_protocol
from .store import atomic, checksum, digest, lock, read_json, write_json
from .scene_store import ROW_DTYPE, read_scene

EVENTS = None


def initialize_worker(queue):
    global EVENTS
    EVENTS = queue
    # Completion files and returned job results are authoritative. Progress events
    # may be dropped during shutdown instead of blocking a worker's queue feeder.
    EVENTS.cancel_join_thread()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def validate_config(config):
    c, e, s = config["sampling"], config["execution"], config["storage"]
    if config.get("objective") == "high_score_256":
        if (
            s["format"] != "high_score_256"
            or s["candidates"] != 256
            or type(s["retain_all_scores"]) is not bool
        ):
            raise ValueError("Expected 256-candidate high-score bank")
        if c["high_pdm_raw"] != 0.95 or c["integer_lattice_m"] != 0.001:
            raise ValueError("Expected raw 0.95 threshold and millimetre request keys")
        budgets = c["query_budgets"]
        if (
            budgets != sorted(set(budgets))
            or budgets[0] != c["max_unique_goals"]
            or budgets[0] < 256
        ):
            raise ValueError("Invalid cumulative query budgets")
        if any(not isinstance(v, int) or v % c["adaptive_batch"] for v in budgets):
            raise ValueError("Budgets must contain complete adaptive batches")
        if s["chunk_candidates"] != c["adaptive_batch"] or c["adaptive_batch"] < 1:
            raise ValueError("Persist complete adaptive batches")
        b, h = c["base_bounds_m"], c["hard_bounds_m"]
        if not (h[0] <= b[0] < 0 < b[1] <= h[1] and h[2] <= b[2] < 0 < b[3] <= h[3]):
            raise ValueError("Invalid search bounds")
        if not 0 < c["exploration_fraction"] <= 1 or any(v <= 0 for v in c["local_scales_m"]):
            raise ValueError("Invalid exploration")
        if e["threads_per_worker"] != 1 or any(
            e[k] < 1 for k in ("workers", "scenes_per_job", "queued_jobs_per_worker")
        ):
            raise ValueError("Invalid execution counts")
        return
    if c["dense_m"] != 0.30 or c["integer_lattice_m"] != 0.05:
        raise ValueError("This profile implements the GT-anchored 0.30 m grid")
    if s["retain_all_scores"] is not True or s["format"] != "scene_structured_npy_float64_labels":
        raise ValueError("All scored candidates must be retained in scene tables")
    for name in ("coarse_m", "offset_probe_m"):
        if c[name] <= 0 or abs(c[name] / c["dense_m"] - round(c[name] / c["dense_m"])) > 1e-8:
            raise ValueError(name + " must be a positive multiple of dense_m")
    for name in ("base_bounds_m", "hard_bounds_m"):
        b = c[name]
        if len(b) != 4 or not b[0] <= 0 <= b[1] or not b[2] <= 0 <= b[3]:
            raise ValueError("Bounds must enclose the GT endpoint")
    b, h = c["base_bounds_m"], c["hard_bounds_m"]
    if not (h[0] <= b[0] <= b[1] <= h[1] and h[2] <= b[2] <= b[3] <= h[3]):
        raise ValueError("Base region must fit inside hard domain")
    for key in ("max_unique_goals", "adaptive_batch", "offset_probe_limit"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if not 0 < c["exploration_fraction"] <= 1 or c["local_radius_m"] <= 0:
        raise ValueError("Invalid adaptive exploration settings")
    if not 0 < c["high_pdm_raw"] <= 1 or not 0 <= c["component_jump_raw"] <= 1:
        raise ValueError("Invalid raw score threshold")
    if s["chunk_candidates"] < 1 or e["threads_per_worker"] != 1:
        raise ValueError("Positive block size and one thread per worker required")
    for key in ("workers", "scenes_per_job", "queued_jobs_per_worker"):
        if e[key] < 1:
            raise ValueError("Invalid execution count: " + key)


def estimate(config, scenes, workers):
    q = config.get("_query_budget", config["sampling"]["max_unique_goals"])
    reference_rate = config["estimate"]["measured_candidates_per_s"]
    # Do not claim linear scaling to unmeasured worker counts.
    rate = reference_rate if workers == config["estimate"]["measured_workers"] else None
    high_bank = config.get("objective") == "high_score_256"
    score_calls = scenes * (q + (257 if high_bank else 5))
    hours = score_calls / rate / 3600 if rate else None
    overhead = config["estimate"]["end_to_end_overhead_fraction"]
    return dict(
        scenes=scenes,
        workers=workers,
        max_candidates_per_scene=q,
        max_candidate_labels=scenes * q,
        max_official_calls_including_audit=score_calls,
        measured_reference_candidates_per_s=reference_rate,
        scoring_hours=hours,
        planning_hours_with_assumed_overhead=[hours * (1 + v) for v in overhead] if hours else None,
        overhead_fraction_is_assumption=overhead,
        trajectory_table_gb=scenes
        * (256 * 168 + 2048 if high_bank else q * ROW_DTYPE.itemsize + 256)
        / 1e9,
        retained_all_scores_gb=(
            scenes * (q * ROW_DTYPE.itemsize + 256) / 1e9
            if high_bank and config["storage"]["retain_all_scores"]
            else 0.0
        ),
        worst_case_staging_gb=scenes * q * ROW_DTYPE.itemsize / 1e9 if high_bank else 0.0,
        required_qualified_candidates_per_scene=256 if high_bank else None,
        qualified_scene_coverage_guaranteed=False if high_bank else None,
        uncached_metric_cache_gb_allowance=scenes
        * config["estimate"]["uncached_scene_allowance_mib"]
        * 2**20
        / 1e9,
        complete_fine_grid_guaranteed=False,
        throughput_requires_matching_worker_count=rate is None,
    )


def load_configuration(path, root):
    config = yaml.safe_load(Path(path).read_text())
    validate_config(config)
    base = yaml.safe_load((root / config["base_protocol"]).read_text())
    validate_protocol(base)
    protocol = copy.deepcopy(base)
    data_root = Path(os.environ.get("OPENSCENE_DATA_ROOT", root / "data"))
    cache_root = Path(os.environ.get("NAVSIM_CACHE_ROOT", root / "data/navsim_cache"))
    scope = protocol["scene_scope"]
    scope["log_root"] = str(data_root / "navsim_logs/trainval")
    scope["metric_cache_root"] = str(cache_root / "train_metric_cache")
    scope["official_cache_root"] = os.environ.get(
        "OFFICIAL_PDM_CACHE", str(cache_root / "official_metric_cache_v1")
    )
    # Source only consumes these constructor, scope and official evaluation fields.
    protocol["sampling"] = config["sampling"]
    protocol["storage"] = config["storage"]
    return config, protocol


def selected_jobs(config, protocol, root, limit):
    items, summary = read_manifest(root / config["manifest_directory"])
    scope = protocol["scene_scope"]
    if len(items) != scope["expected_total"] or any(
        summary["splits"][s]["scenes"] != scope["expected_" + s] for s in ("train", "val")
    ):
        raise ValueError("Manifest differs from the agreed train/val counts")
    # build_jobs also checks duplicate tokens and log split overlap before selection.
    return build_jobs(
        items,
        dict(
            splits=["train", "val"],
            max_scenes=limit,
            selection_seed=config["execution"]["selection_seed"],
        ),
        config["execution"]["scenes_per_job"],
    )


def run_job(job, output, protocol, config, contract_id):
    if config.get("objective") == "high_score_256":
        from .high_score_scene import generate
    else:
        from .production_scene import generate
    tokens = [r["token"] for r in job["scenes"]]
    results = []
    for r in records([job]):
        try:
            result = generate(r, tokens, output, protocol, config, contract_id)
            event = dict(
                token=r["token"],
                split=r["split"],
                passed=result["passed"],
                status=result.get("status", "complete"),
                queries=result.get("queries", result["candidates"]),
                candidates=result["candidates"],
                completed_unix=result["completed_unix"],
                full_scores_available=(
                    Path(output) / "scenes" / r["split"] / r["token"] / "ALL_SCORES.json"
                ).exists(),
            )
        except OSError:
            raise  # Stop the shared queue on disk/cache I/O failure.
        except Exception:
            event = dict(
                token=r["token"], split=r["split"], passed=False, traceback=traceback.format_exc()
            )
            write_json(Path(output) / "scenes" / r["split"] / r["token"] / "ERROR.json", event)
        results.append(event)
        if EVENTS is not None:
            EVENTS.put(event)
    return results


def scan_completed(
    output,
    selected,
    verify=True,
    contract_id=None,
    retain_all_scores=False,
    scan_workers=1,
    report=None,
    full_only=False,
):
    def inspect(r):
        directory = output / "scenes" / r["split"] / r["token"]
        if (directory / "COMPLETE.json").exists():
            if (
                contract_id is not None
                and read_json(directory / "contract.json")["run_contract"] != contract_id
            ):
                raise ValueError("Completed scene run contract differs")
            high_bank = read_json(directory / "COMPLETE.json").get("format") == "high_score_256"
            if verify or high_bank:
                reader = read_scene
                if high_bank:
                    from .high_score_bank import read_scene as reader
                with lock(directory):
                    identity, rows = reader(directory, verify=True)
                    if any(
                        identity[k] != r[k]
                        for k in ("token", "split", "log_name", "metric_cache_relative_path")
                    ):
                        raise ValueError("Stored scene identity differs from manifest")
                    del rows
                    full = (directory / "ALL_SCORES.json").exists()
                    if full:
                        from .retained_bank import read_full_scene

                        read_full_scene(directory)
                    if high_bank:
                        from .high_score_scene import cleanup_staging

                        cleanup_staging(directory, retain_all_scores)
            item = read_json(directory / "COMPLETE.json")
            if not item.get("passed"):
                raise ValueError("Invalid completion marker")
            full = (directory / "ALL_SCORES.json").exists()
            if full_only and not full:
                return None
            return dict(
                token=r["token"],
                split=r["split"],
                passed=True,
                candidates=item["candidates"],
                queries=item.get("queries", item["candidates"]),
                full_scores_available=full,
            )
        return None

    completed, scanned = {}, 0

    def receive(item):
        nonlocal scanned
        scanned += 1
        if item is not None:
            completed[item["token"]] = item
        if report:
            report(scanned, completed)

    if scan_workers == 1:
        for r in selected:
            receive(inspect(r))
    else:
        iterator = iter(selected)
        with ThreadPoolExecutor(max_workers=scan_workers) as pool:
            pending = {}
            for _ in range(min(len(selected), scan_workers * 2)):
                r = next(iterator)
                pending[pool.submit(inspect, r)] = r
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    receive(future.result())
                    r = next(iterator, None)
                    if r is not None:
                        pending[pool.submit(inspect, r)] = r
    return completed


def progress(
    completed, failures, initial_count, total, start, workers, state="running", shortfalls=None
):
    shortfalls = shortfalls or {}
    elapsed = time.monotonic() - start
    fresh = len(completed) + len(shortfalls) - initial_count
    rate = fresh / elapsed if elapsed > 0 else 0
    return dict(
        state=state,
        completed_scenes=len(completed),
        total_scenes=total,
        failed_scenes=len(failures),
        complete_candidates=sum(r["candidates"] for r in completed.values()),
        shortfall_scenes=len(shortfalls),
        processed_scenes=len(completed) + len(shortfalls),
        full_score_scenes=sum(
            bool(r.get("full_scores_available"))
            for r in list(completed.values()) + list(shortfalls.values())
        ),
        full_score_candidates=sum(
            r.get("queries", 0)
            for r in list(completed.values()) + list(shortfalls.values())
            if r.get("full_scores_available")
        ),
        eligible_scene_fraction=len(completed) / total if total else 0.0,
        invocation_completed_scenes=fresh,
        invocation_elapsed_s=elapsed,
        workers=workers,
        scenes_per_hour=rate * 3600,
        remaining_hours=(
            (total - len(completed) - len(shortfalls)) / rate / 3600 if fresh >= workers else None
        ),
        eta_basis="completed scenes per wall-clock second including loading, scoring, audit and writing",
        updated_unix=time.time(),
    )


def execute(jobs, output, protocol, config, contract_id, completed, workers):
    start = time.monotonic()
    total = sum(len(j["scenes"]) for j in jobs)
    initial_count, failures, shortfalls = len(completed), {}, {}
    prefix = "backfill_" if config.get("_backfill_full_scores") else ""
    marker_prefix = "BACKFILL_" if prefix else ""
    pending_jobs = []
    for job in jobs:
        remaining = [r for r in job["scenes"] if r["token"] not in completed]
        if remaining:
            pending_jobs.append(dict(job, scenes=remaining))
    # Larger jobs first reduces end-of-run imbalance without changing per-scene samples.
    pending_jobs.sort(key=lambda j: (-len(j["scenes"]), j["job_id"]))
    context = mp.get_context("spawn")
    events = context.Queue()
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=initialize_worker, initargs=(events,)
    )
    pending, iterator = {}, iter(pending_jobs)

    def stop(signum, frame):
        raise KeyboardInterrupt("Resume with the same command; committed rows are retained")

    previous = signal.signal(signal.SIGTERM, stop)
    last_report = 0.0

    def receive(event):
        if event["passed"]:
            completed[event["token"]] = event
            failures.pop(event["token"], None)
            shortfalls.pop(event["token"], None)
        elif event.get("status") == "shortfall":
            shortfalls[event["token"]] = event
            failures.pop(event["token"], None)
        else:
            failures[event["token"]] = event

    def report(state="running"):
        value = progress(
            completed, failures, initial_count, total, start, workers, state, shortfalls
        )
        value["phase"] = "backfill" if prefix else "high_score_generation"
        write_json(output / (prefix + "status.json"), value)
        print(json.dumps(value), flush=True)
        return value

    try:
        for _ in range(
            min(len(pending_jobs), workers * config["execution"]["queued_jobs_per_worker"])
        ):
            job = next(iterator)
            pending[pool.submit(run_job, job, str(output), protocol, config, contract_id)] = job[
                "job_id"
            ]
        while pending:
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            while True:
                try:
                    receive(events.get_nowait())
                except Empty:
                    break
            for future in done:
                pending.pop(future)
                for event in future.result():
                    receive(event)
                job = next(iterator, None)
                if job is not None:
                    pending[
                        pool.submit(run_job, job, str(output), protocol, config, contract_id)
                    ] = job["job_id"]
            if time.monotonic() - last_report >= config["execution"]["progress_interval_s"]:
                report()
                last_report = time.monotonic()
        pool.shutdown(wait=True)
        if failures:
            report("failed")
            raise RuntimeError(
                f"{len(failures)} scenes failed; inspect per-scene ERROR.json and resume"
            )
        if len(completed) + len(shortfalls) != total:
            raise RuntimeError("Completed scene count differs from requested count")
        result = report("partial" if shortfalls else "complete")
        result["passed"] = not shortfalls
        if shortfalls:
            result["shortfalls"] = list(shortfalls.values())
            write_json(output / (marker_prefix + "PARTIAL.json"), result)
        else:
            write_json(output / (marker_prefix + "COMPLETE.json"), result)
            (output / (marker_prefix + "PARTIAL.json")).unlink(missing_ok=True)
        return result
    except BaseException:
        for process in list((getattr(pool, "_processes", None) or {}).values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        report("interrupted_or_failed")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        events.close()


def source_contract(root, protocol, config, selected):
    files = [
        root / "feedback/src/drivesta_feedback/batch" / (n + ".py")
        for n in (
            "production",
            "production_scene",
            "budget_sampler",
            "scene_store",
            "source",
            "store",
            "cache",
            "plan",
            "runner",
        )
    ]
    if config.get("objective") == "high_score_256":
        files += [
            root / "feedback/src/drivesta_feedback/batch" / (n + ".py")
            for n in ("high_score_bank", "high_score_sampler", "high_score_scene", "retained_bank")
        ]
    files += [
        root / p
        for p in (
            "navsim/tools/trajectory_mining/path_endpoint_surface.py",
            "navsim/evaluate/pdm_score.py",
            "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
            "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
            "navsim/planning/metric_caching/metric_cache_processor.py",
        )
    ]
    return dict(
        version=config["version"],
        protocol=protocol,
        selected_digest=digest(selected),
        scene_count=len(selected),
        sampling=config["sampling"],
        storage=config["storage"],
        implementation={str(p.relative_to(root)): checksum(p) for p in files},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("estimate", "run", "status", "backfill"))
    parser.add_argument(
        "--config", type=Path, default=Path("feedback/configs/navtrainval_030.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--scan-workers", type=int, default=32, help="Bounded threads for recovery file checks"
    )
    parser.add_argument(
        "--upgrade-retention",
        action="store_true",
        help="Explicit retention-only upgrade of an existing run",
    )
    parser.add_argument(
        "--backfill-after-complete",
        action="store_true",
        help="Backfill legacy full tables only after all high banks pass",
    )
    parser.add_argument(
        "--query-budget",
        type=int,
        help="High-score profile: cumulative budget for unfinished scenes",
    )
    parser.add_argument(
        "--limit", type=int, help="Optional log-spread subset; use a separate output directory"
    )
    args = parser.parse_args()
    root = project_root()
    config, protocol = load_configuration(root / args.config, root)
    if args.query_budget is not None:
        if (
            config.get("objective") != "high_score_256"
            or args.query_budget not in config["sampling"]["query_budgets"]
        ):
            parser.error("query-budget must be a configured high-score cumulative budget")
        config["_query_budget"] = args.query_budget
    output = root / (args.output or config["output"])
    workers = config["execution"]["workers"] if args.workers is None else args.workers
    if workers < 1 or args.scan_workers < 1:
        parser.error("workers must be positive")
    if args.action == "status":
        print(json.dumps(read_json(output / "status.json"), indent=2))
        return
    jobs = selected_jobs(config, protocol, root, args.limit)
    selected = records(jobs)
    costs = estimate(config, len(selected), workers)
    print(json.dumps(costs, indent=2), flush=True)
    if args.action == "estimate":
        return
    contract = source_contract(root, protocol, config, selected)
    with lock(output):
        from .retained_bank import resolve_run_contract

        identity_contract = resolve_run_contract(output, contract, args.upgrade_retention)
        # Existing partial scene tables carry this numerical protocol exactly.
        protocol = identity_contract["protocol"]
        contract_id = digest(identity_contract)
        followup_path = output / "followup.json"
        followup = read_json(followup_path) if followup_path.exists() else {}
        followup_requested = args.backfill_after_complete or followup.get("requested", False)
        if args.backfill_after_complete:
            write_json(
                followup_path,
                dict(
                    requested=True,
                    action="backfill",
                    prerequisite="all top256 banks complete",
                    state="waiting_for_high_score_completion",
                ),
            )
        manifest_path = output / "scenes.jsonl"
        encoded = "".join(json.dumps(r) + "\n" for r in selected).encode()
        if manifest_path.exists() and manifest_path.read_bytes() != encoded:
            raise ValueError("Run manifest mismatch")
        if not manifest_path.exists():
            atomic(manifest_path, lambda f: f.write(encoded))
        if (args.action == "backfill" or followup_requested) and not config["storage"][
            "retain_all_scores"
        ]:
            parser.error("Backfill requires retain_all_scores: true")
        if args.action == "backfill":
            require_high_banks_complete(output, len(selected))
            config["_backfill_full_scores"] = True
        recovery_start, last_report = time.monotonic(), [0.0]

        def report_recovery(scanned, completed):
            now = time.monotonic()
            if now - last_report[0] >= 10 or scanned == len(selected):
                value = dict(
                    state="verifying" if scanned < len(selected) else "verified",
                    scanned_scenes=scanned,
                    total_scenes=len(selected),
                    validated_banks=len(completed),
                    elapsed_s=now - recovery_start,
                    workers=args.scan_workers,
                    updated_unix=time.time(),
                )
                write_json(output / "recovery.json", value)
                print(json.dumps(dict(recovery=value)), flush=True)
                last_report[0] = now

        completed = scan_completed(
            output,
            selected,
            contract_id=contract_id,
            retain_all_scores=config["storage"]["retain_all_scores"],
            scan_workers=args.scan_workers,
            report=report_recovery,
            full_only=args.action == "backfill",
        )
        remaining = len(selected) - len(completed)
        q = config.get("_query_budget", config["sampling"]["max_unique_goals"])
        needed = remaining * (q * ROW_DTYPE.itemsize + 50000)
        cache_allowance = remaining * config["estimate"]["uncached_scene_allowance_mib"] * 2**20
        # Conservatively reserve both tables and new official caches on this installation's filesystem.
        reserve = config["execution"]["minimum_free_gib"] * 2**30
        if shutil.disk_usage(output).free < needed + cache_allowance + reserve:
            raise OSError(
                28,
                "Insufficient free space for remaining scene tables, official caches and reserve",
            )
        write_json(output / "estimate.json", costs)
        write_json(
            output / "execution.json",
            dict(
                workers=workers,
                threads_per_worker=1,
                host=os.uname().nodename,
                started_unix=time.time(),
                resume_scenes=len(completed),
                config=config,
                source_contract=contract,
                backfill_after_complete=followup_requested,
                selected_split_counts=dict(Counter(r["split"] for r in selected)),
            ),
        )
        result = execute(jobs, output, protocol, config, contract_id, completed, workers)
        if result and not result["passed"]:
            if followup_requested:
                write_json(
                    followup_path,
                    dict(
                        requested=True,
                        action="backfill",
                        prerequisite="all top256 banks complete",
                        state="waiting_for_shortfalls",
                        shortfall_scenes=result["shortfall_scenes"],
                    ),
                )
            raise SystemExit(2)  # Finished the requested budget with explicit shortfalls.
        if args.action == "backfill" and followup_requested:
            write_json(followup_path, dict(requested=True, action="backfill", state="complete"))
        if followup_requested and args.action == "run":
            require_high_banks_complete(output, len(selected))
            full = {k: v for k, v in completed.items() if v.get("full_scores_available")}
            config["_backfill_full_scores"] = True
            write_json(followup_path, dict(requested=True, action="backfill", state="running"))
            execute(jobs, output, protocol, config, contract_id, full, workers)
            write_json(followup_path, dict(requested=True, action="backfill", state="complete"))


def require_high_banks_complete(output, count):
    path = Path(output) / "COMPLETE.json"
    if not path.exists():
        raise ValueError("Backfill waits for every scene to have a verified top256 bank")
    info = read_json(path)
    if (
        not info.get("passed")
        or info.get("completed_scenes") != count
        or info.get("shortfall_scenes", 0)
    ):
        raise ValueError("High-score scene coverage is incomplete; finish it before backfill")


if __name__ == "__main__":
    main()
