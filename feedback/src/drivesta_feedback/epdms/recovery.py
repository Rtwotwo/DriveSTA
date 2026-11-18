"""Resume bounded NAVSIM recovery and publish only fully qualified scene banks."""

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
from pathlib import Path
import json, csv, fcntl, shutil, time, traceback, queue, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from .dataset import read_scene, read_target_pairs
from .recovery_search import search
from .recovery_store import (
    workspace,
    resume_transactions,
    verify_result,
    publish,
    REVISION,
    snapshot_original,
)
from ..batch.store import write_json, checksum

EVENTS = None


def initialize(events):
    global EVENTS
    EVENTS = events


def trial_folder(root, token):
    pilot = workspace(root) / "pilot" / "scenes" / token
    return pilot if (pilot / "result.json").exists() else workspace(root) / "scenes" / token


def process_log(task):
    root, rows, rounds = task
    root = Path(root)
    results = []
    for i, row in enumerate(rows):
        if row["selected_count"] == 256:
            continue
        work = trial_folder(root, row["token"])
        try:
            result = search(root, row, work, rounds=rounds)
            result = dict(result)
            if result["recovered"]:
                valid = work / "validation.json"
                if not valid.exists() or json.loads(valid.read_text())[
                    "best_file_sha256"
                ] != checksum(work / "best.npy"):
                    verify_result(root, row, work, independent_count=8)
                next_row = rows[i + 1] if i + 1 < len(rows) else None
                info = publish(root, row, next_row, work)
                # Exercise the same context reader used by paired training.
                paired = read_target_pairs(root, row["split"], row["token"])
                if len(paired["current"]["query_indices"]) != 256:
                    raise ValueError("Post-publication target count")
                result["published"] = True
                result["scene_info"] = info
                if next_row is not None:
                    result["successor_info"] = read_scene(
                        root / "scenes" / next_row["split"] / next_row["token"], verify=False
                    )[0]
            else:
                result["published"] = False
            result["error"] = None
        except Exception:
            result = dict(
                token=row["token"],
                split=row["split"],
                published=False,
                error=traceback.format_exc(),
            )
            work.mkdir(parents=True, exist_ok=True)
            write_json(work / "error.json", result)
        results.append(result)
        if EVENTS is not None:
            EVENTS.put(result)
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in [
                        "token",
                        "original_count",
                        "selected_count",
                        "published",
                        "queries",
                        "error",
                    ]
                }
            ),
            flush=True,
        )
    write_json(
        workspace(root) / "log_results" / (rows[0]["log_name"] + ".json"), dict(results=results)
    )
    return len(results)


def write_state(root, original, results, started, final=False, job_errors=None):
    root = Path(root)
    base = {r["token"]: dict(r) for r in original["scenes"]}
    for result in results.values():
        for key in ("scene_info", "successor_info"):
            if key in result:
                value = result[key]
                # A later recovered successor beats a prior metadata-only record.
                if base[value["token"]].get("recovery_revision") and not value.get(
                    "recovery_revision"
                ):
                    continue
                base[value["token"]] = value
    rows = list(base.values())
    remaining = sorted(r["token"] for r in rows if r["selected_count"] < 256)
    errors = [
        dict(token=r["token"], traceback=r["error"]) for r in results.values() if r.get("error")
    ]
    errors += job_errors or []
    recovered = sum(bool(r.get("published")) for r in results.values())
    status = dict(
        phase="recovery_finished" if final else "recovery_running",
        updated_unix=time.time(),
        processed=103288,
        expected=103288,
        trajectory_records=103288 * 3072,
        high256=sum(r["selected_count"] == 256 for r in rows),
        targets=sum(r["selected_count"] for r in rows),
        shortfall_processed=len(remaining),
        recovery_processed=len(results),
        recovery_expected=31565,
        recovered_this_round=recovered,
        recovery_elapsed_s=time.time() - started,
        errors=errors,
        workers_requested=96,
        complete=final and not errors and len(results) == 31565,
    )
    write_json(root / "status.json", status)
    write_json(workspace(root) / "status.json", status)
    # Durable progress prevents reported totals from falling back on resume.
    compact = {t: {k: v for k, v in r.items() if k not in ("rounds",)} for t, r in results.items()}
    write_json(workspace(root) / "progress_records.json", compact)
    p = root / "shortfall_tokens.txt.tmp"
    p.write_text("".join(t + "\n" for t in remaining))
    p.replace(root / "shortfall_tokens.txt")
    p = root / "shortfall_details.csv.tmp"
    with p.open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "token",
                "split",
                "selected_count",
                "missing_count",
                "best_epdms",
                "recovery_attempted",
                "recovery_found",
                "first_funnel_below256",
            ],
        )
        w.writeheader()
        for token in remaining:
            row = base[token]
            attempt = results.get(token, {})
            first = next(
                (
                    k
                    for k in [
                        "raw_score_gt95",
                        "raw_safety_gt95",
                        "physical_gt95",
                        "tracking_gt95",
                        "space_gt95",
                    ]
                    if row["funnel"].get(k, 0) < 256
                ),
                "deduplication",
            )
            w.writerow(
                dict(
                    token=token,
                    split=row["split"],
                    selected_count=row["selected_count"],
                    missing_count=256 - row["selected_count"],
                    best_epdms=row["best_epdms"],
                    recovery_attempted=token in results,
                    recovery_found=attempt.get("selected_count", ""),
                    first_funnel_below256=first,
                )
            )
    p.replace(root / "shortfall_details.csv")
    if final:
        summary = dict(
            original,
            scenes=rows,
            high256=status["high256"],
            targets=status["targets"],
            with_targets=sum(r["selected_count"] > 0 for r in rows),
            ec_pass=sum(r["ec_pass_count"] or 0 for r in rows),
            recovery=status,
            summary_updated_unix=time.time(),
        )
        write_json(root / "summary.json", summary)
        groups = {}
        for row in rows:
            groups.setdefault(row["log_name"], []).append(row)
        for log, records in groups.items():
            write_json(root / "logs" / (log + ".json"), dict(complete=True, summaries=records))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--rounds", type=int, default=8)
    a = p.parse_args()
    root = a.root.resolve()
    work = workspace(root)
    work.mkdir(parents=True, exist_ok=True)
    if not 1 <= a.workers <= 96 or not 1 <= a.rounds <= 8:
        raise ValueError("Invalid bounded production settings")
    handle = (root / ".writer.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resume_transactions(root)
    original_file = snapshot_original(root)
    original = json.loads(original_file.read_text())
    if (
        not original["complete"]
        or original["processed"] != 103288
        or original["high256"] != 71723
        or original["errors"]
    ):
        raise ValueError("Unexpected original corpus")
    config = dict(
        revision=REVISION,
        workers=a.workers,
        rounds=a.rounds,
        original_summary_sha256=checksum(original_file),
        max_new_queries_per_scene=3072 * a.rounds,
        strict_threshold=0.95,
        criterion="all original physical, raw safety, tracking and spatial conditions unchanged",
        context="immutable original same-family predecessor, before scoring; no rematching",
        independent_replay_per_recovered_scene=8,
        all_selected_freshly_replayed=True,
        code_sha256={p.name: checksum(p) for p in Path(__file__).parent.glob("*.py")},
        reused_pilot="Stored six-round pilot query journals are retained verbatim; incomplete scenes receive two additional rounds.",
    )
    config_file = work / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Recovery configuration changed")
    write_json(config_file, config)
    groups = {}
    for row in original["scenes"]:
        groups.setdefault(row["log_name"], []).append(row)
    jobs = []
    for rows in groups.values():
        rows.sort(key=lambda r: r["start_time"])
        if any(r["selected_count"] < 256 for r in rows):
            jobs.append((str(root), rows, a.rounds))
    jobs.sort(key=lambda j: -sum(r["selected_count"] < 256 for r in j[1]))
    saved = work / "progress_records.json"
    results = json.loads(saved.read_text()) if saved.exists() else {}
    os.nice(10)
    context = mp.get_context("fork")
    events = context.Queue()
    job_errors = []
    started = time.time()
    last = 0
    with ProcessPoolExecutor(
        max_workers=a.workers, mp_context=context, initializer=initialize, initargs=(events,)
    ) as pool:
        pending = {pool.submit(process_log, j): j[1][0]["log_name"] for j in jobs}
        while pending:
            done, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            while True:
                try:
                    r = events.get_nowait()
                    results[r["token"]] = r
                except queue.Empty:
                    break
            for f in done:
                name = pending.pop(f)
                try:
                    f.result()
                except Exception:
                    job_errors.append(dict(log_name=name, traceback=traceback.format_exc()))
            if time.time() - last > 45:
                s = write_state(root, original, results, started, job_errors=job_errors)
                last = time.time()
                print(json.dumps(s), flush=True)
        # All worker queue writers have exited after the pool context.
    while True:
        try:
            r = events.get_nowait()
            results[r["token"]] = r
        except queue.Empty:
            break
    # Queue delivery may lag a completed future. Log results are authoritative.
    for f in (work / "log_results").glob("*.json"):
        for r in json.loads(f.read_text())["results"]:
            results[r["token"]] = r
    status = write_state(root, original, results, started, final=True, job_errors=job_errors)
    write_json(work / "FINAL.json", status)
    print(json.dumps(status), flush=True)
    if status["errors"]:
        raise RuntimeError("Recovery has errors; preserve journals and resume")


if __name__ == "__main__":
    main()
