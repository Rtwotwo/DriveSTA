"""Bounded log-job scheduler for preflight, generation, audit and packing."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import traceback

from drivesta_feedback.paths import project_root
from .store import checksum, digest, read_json, write_json, lock


def validate_protocol(protocol):
    """Reject unsupported config variants instead of recording unused settings."""
    required = {
        ("constructor", "reference"): "expert_gt",
        ("constructor", "parents_per_goal"): 1,
        ("constructor", "factor"): 1.0,
        ("constructor", "goal_shrink"): False,
        ("constructor", "num_poses"): 8,
        ("constructor", "interval_s"): 0.5,
        ("labeling", "protocol"): "navsim_v1_official_single_candidate",
        ("labeling", "simulator_poses"): 40,
        ("labeling", "simulator_interval_s"): 0.1,
        ("labeling", "evaluation_group"): "reference_plus_one_candidate",
        ("sampling", "integer_lattice_m"): 0.05,
        ("sampling", "dense_m"): 0.1,
        ("sampling", "boundary_m"): 0.05,
    }
    for (section, key), value in required.items():
        if protocol[section][key] != value:
            raise ValueError(f"Unsupported protocol setting: {section}.{key}")
    if protocol["storage"]["chunk_candidates"] < 1 or protocol["sampling"]["max_unique_goals"] < 1:
        raise ValueError("Chunk and candidate budgets must be positive")


def records(jobs):
    return [dict(r, split=j["split"], log_name=j["log_name"]) for j in jobs for r in j["scenes"]]


def task(job, action, output, protocol):
    from .source import load_scene
    from .worker import generate_scene
    from .audit import audit_scene

    tokens = [r["token"] for r in job["scenes"]]
    results = []
    for r in records([job]):
        directory = Path(output) / "scenes" / r["split"] / r["token"]
        try:
            if action == "preflight":
                from .source import Evaluator, construct
                import numpy as np

                _, gt, cache, meta = load_scene(r, tokens, protocol)
                evaluator = Evaluator(cache)
                gt_labels = evaluator.score(gt)
                probe, error = construct(gt, np.array([[0, 0]], dtype=np.int32), protocol)
                probe_labels = evaluator.score(probe[0])
                result = dict(
                    token=r["token"],
                    passed=True,
                    timestamp_us=meta["timestamp_us"],
                    pose_error=meta["pose_error"],
                    gt_pdms=float(gt_labels[-1] * 100),
                    zero_offset_pdms=float(probe_labels[-1] * 100),
                    endpoint_error_m=error,
                    official_cache_path=meta["official_cache_path"],
                )
            elif action == "generate":
                result = generate_scene(r, tokens, directory, protocol)
                # Reconcile a crash between publishing COMPLETE and updating progress.
                write_json(
                    directory / "progress.json",
                    dict(
                        status="complete", candidates=result["candidates"], updated_unix=time.time()
                    ),
                )
            else:
                result = audit_scene(directory, protocol)
            results.append(result)
            print(
                json.dumps(
                    dict(
                        event=action,
                        token=r["token"],
                        candidates=result.get("candidates"),
                        pid=os.getpid(),
                    )
                ),
                flush=True,
            )
        except OSError:
            # Global storage failures stop dispatch; an exception is never converted to a zero label.
            raise
        except Exception:
            error = dict(
                token=r["token"], action=action, traceback=traceback.format_exc(), failed=True
            )
            write_json(Path(output) / "errors" / f'{r["token"]}_{action}.json', error)
            results.append(error)
    return results


def execute(jobs, action, output, protocol, workers):
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    pending = {}
    iterator = iter(jobs)
    results = []

    def stop(signum, frame):
        raise KeyboardInterrupt("Stopping scheduler; committed blocks are recoverable")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        for _ in range(min(len(jobs), 2 * workers)):
            job = next(iterator)
            pending[pool.submit(task, job, action, str(output), protocol)] = job["job_id"]
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=30)
            if not done:
                print(
                    json.dumps(
                        dict(
                            event="heartbeat",
                            action=action,
                            completed=len(results),
                            active_jobs=len(pending),
                        )
                    ),
                    flush=True,
                )
            for future in done:
                pending.pop(future)
                results.extend(future.result())
                write_json(
                    output / (action + "_progress.json"),
                    dict(
                        completed=len(results),
                        failed=sum(bool(r.get("failed")) for r in results),
                        updated_unix=time.time(),
                    ),
                )
                job = next(iterator, None)
                if job is not None:
                    pending[pool.submit(task, job, action, str(output), protocol)] = job["job_id"]
        pool.shutdown(wait=True)
    except BaseException:
        for process in list(pool._processes.values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    return sorted(results, key=lambda r: r["token"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "generate", "audit", "pack", "all"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-root", default="data/navsim_logs/trainval")
    parser.add_argument("--pack-output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = project_root()
    jobs = [
        json.loads(line)
        for line in (args.plan / "jobs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = records(jobs)
    if len({r["token"] for r in selected}) != len(selected):
        raise ValueError("Duplicate planned scene")
    manifest = root / "feedback/data/dense_gt_v1/manifests/navtrainval_v1/scenes.jsonl"
    wanted = {r["token"]: r for r in selected}
    found = set()
    with open(manifest) as f:
        for line in f:
            r = json.loads(line)
            if r["token"] in wanted:
                for key in ("split", "log_name", "metric_cache_relative_path"):
                    if r[key] != wanted[r["token"]][key]:
                        raise ValueError("Plan/manifest mismatch")
                found.add(r["token"])
    if found != set(wanted):
        raise ValueError("Unknown token or test scene in plan")
    protocol = read_json(args.plan / "dataset_protocol.json")
    validate_protocol(protocol)
    protocol["scene_scope"]["log_root"] = args.log_root
    protocol["scene_scope"][
        "official_cache_root"
    ] = "feedback/data/dense_gt_v1/official_metric_cache_v1"
    protocol["implementation_version"] = "batch_v1"
    source_paths = [
        root / "feedback/src/drivesta_feedback/batch" / f"{name}.py"
        for name in ("sampler", "source", "worker", "store", "cache")
    ]
    source_paths += [
        root / "navsim/tools/trajectory_mining/path_endpoint_surface.py",
        root / "navsim/evaluate/pdm_score.py",
        root / "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        root / "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
        root / "navsim/planning/metric_caching/metric_cache_processor.py",
    ]
    protocol["implementation_files"] = {str(p.relative_to(root)): checksum(p) for p in source_paths}
    contract = dict(
        schema_version=1, jobs_digest=digest(jobs), scene_count=len(selected), protocol=protocol
    )
    output = args.output
    with lock(output):
        if (output / "run.json").exists() and read_json(output / "run.json") != contract:
            raise ValueError(
                "Run configuration/source differs; use the original source or a separate run directory"
            )
        write_json(output / "run.json", contract)
        # Every startup checks durable write, fsync, rename and readback on the output filesystem.
        write_json(output / "storage_probe.json", dict(pid=os.getpid(), time=time.time()))
        write_json(
            output / "execution.json",
            dict(
                workers=args.workers,
                parallel_unit="scene",
                threads_per_worker=1,
                device="cpu",
                started=time.time(),
            ),
        )
        actions = (
            ["preflight", "generate", "audit", "pack"] if args.action == "all" else [args.action]
        )
        for action in actions:
            start = time.perf_counter()
            if action == "pack":
                from .pack import pack_dataset

                result = pack_dataset(output, args.pack_output or output / "packed")
            else:
                if action == "generate" and not read_json(output / "preflight.json")["passed"]:
                    raise ValueError("Successful preflight required")
                rows = execute(jobs, action, output, protocol, args.workers)
                failed = sum(bool(r.get("failed")) for r in rows)
                result = dict(
                    passed=failed == 0,
                    failed=failed,
                    scene_count=len(rows),
                    scenes=rows,
                    elapsed_s=time.perf_counter() - start,
                )
                write_json(output / (action + ".json"), result)
                if failed:
                    raise RuntimeError(
                        f"{action}: {failed} scenes failed; rerun the same command after resolving errors"
                    )
                if action == "generate":
                    write_json(output / "COMPLETE.json", result)
            print(
                json.dumps(
                    dict(
                        event="finished",
                        action=action,
                        elapsed_s=time.perf_counter() - start,
                        scenes=result.get("scene_count", result.get("scenes")),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
