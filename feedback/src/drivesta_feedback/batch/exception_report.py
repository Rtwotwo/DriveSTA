"""Read published scene exceptions without interrupting dataset generation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

from .store import atomic, write_json


def optional_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def inspect_scene(root, record):
    directory = root / "scenes" / record["split"] / record["token"]
    identity = {k: record[k] for k in ("token", "split", "log_name")}
    shortfall, errors = None, []
    for name in ("SHORTFALL.json", "ERROR.json"):
        try:
            marker = optional_json(directory / name)
            if marker is None:
                continue
            if any(marker.get(k, record[k]) != record[k] for k in ("token", "split")):
                raise ValueError("Exception marker identity differs from scene manifest")
            complete = optional_json(directory / "COMPLETE.json")
            high_complete = bool(complete and complete.get("passed"))
            if name == "ERROR.json":
                # Errors during full-table replay matter even when top256 is complete.
                trace = marker.get("traceback", marker.get("error", ""))
                errors.append(
                    dict(
                        identity,
                        category="generation_error",
                        high_bank_complete=high_complete,
                        detail=str(trace).strip(),
                        marker_path=str(directory / name),
                    )
                )
                continue
            if high_complete:
                continue  # A completed bank supersedes an older shortfall marker.
            count = int(marker["candidates"])
            if not 0 <= count < 256 or int(marker["missing_candidates"]) != 256 - count:
                raise ValueError("Invalid shortfall count")
            state = optional_json(directory / "state.json") or {}
            queries = int(marker["queries"])
            committed = state.get("count", queries)
            pending = state.get("pending") is not None
            # Recheck publication after reading state: a worker may just have finished.
            complete = optional_json(directory / "COMPLETE.json")
            if complete and complete.get("passed"):
                continue
            best = marker.get("best_pdms")
            shortfall = dict(
                identity,
                qualified_unique=count,
                missing=256 - count,
                scored_queries=queries,
                qualified_queries=marker.get("qualified_queries"),
                best_pdms=best,
                min_selected_pdms=marker.get("min_selected_pdms"),
                committed_queries_now=committed,
                status=(
                    "additional_queries_in_progress"
                    if pending or committed > queries
                    else "shortfall"
                ),
                reason=(
                    "no_candidate_reaches_95"
                    if best is not None and best < 95
                    else "fewer_than_256_distinct_eligible"
                ),
                full_scores_published=(directory / "ALL_SCORES.json").exists(),
                marker_updated_unix=marker.get("completed_unix"),
                observed_unix=time.time(),
                marker_path=str(directory / name),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(
                dict(
                    identity,
                    category="report_read_error",
                    high_bank_complete=None,
                    detail=str(exc),
                    marker_path=str(directory / name),
                )
            )
    return shortfall, errors


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8-sig")
    atomic(path, lambda f: f.write(payload))


def collect(root, destination, workers=16, verbose=True):
    root, destination = Path(root), Path(destination)
    started = time.time()
    manifest = [
        json.loads(line)
        for line in (root / "scenes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    seen = set()
    for record in manifest:
        token = record["token"]
        if (
            record["split"] not in ("train", "val")
            or not token
            or any(c not in "0123456789abcdef" for c in token)
        ):
            raise ValueError("Invalid manifest token or split")
        if token in seen:
            raise ValueError("Duplicate scene token")
        seen.add(token)
    shortfalls, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bound outstanding metadata requests on the shared production filesystem.
        for offset in range(0, len(manifest), 512):
            batch = manifest[offset : offset + 512]
            for shortfall, failures in pool.map(lambda r: inspect_scene(root, r), batch):
                if shortfall:
                    shortfalls.append(shortfall)
                errors.extend(failures)
            if verbose and (offset % 10240 == 0 or offset + len(batch) == len(manifest)):
                print(
                    json.dumps(
                        dict(
                            scanned=min(offset + 512, len(manifest)),
                            total=len(manifest),
                            shortfalls=len(shortfalls),
                            errors=len(errors),
                        )
                    ),
                    flush=True,
                )
    shortfalls.sort(
        key=lambda r: (
            -r["missing"],
            r["best_pdms"] if r["best_pdms"] is not None else -1,
            r["token"],
        )
    )
    errors.sort(key=lambda r: (r["category"], r["token"]))
    counts = Counter(r["category"] for r in errors)
    summary = dict(
        dataset_root=str(root),
        started_unix=started,
        finished_unix=time.time(),
        scanned_scenes=len(manifest),
        shortfall_scenes=len(shortfalls),
        total_missing_qualified_trajectories=sum(r["missing"] for r in shortfalls),
        no_qualified_trajectory_scenes=sum(r["qualified_unique"] == 0 for r in shortfalls),
        below_95_best_score_scenes=sum(
            r["reason"] == "no_candidate_reaches_95" for r in shortfalls
        ),
        shortfall_split_counts=dict(Counter(r["split"] for r in shortfalls)),
        generation_error_scenes=len(
            {r["token"] for r in errors if r["category"] == "generation_error"}
        ),
        report_read_errors=counts["report_read_error"],
        definition="Published shortfall without a passed top256 COMPLETE marker; unprocessed scenes are excluded.",
        live_snapshot=True,
        note="Scenes are observed during the stated scan interval. Additional queries and later completions can change this list.",
    )
    destination.mkdir(parents=True, exist_ok=True)
    short_fields = [
        "token",
        "split",
        "log_name",
        "qualified_unique",
        "missing",
        "scored_queries",
        "qualified_queries",
        "best_pdms",
        "min_selected_pdms",
        "committed_queries_now",
        "status",
        "reason",
        "full_scores_published",
        "marker_updated_unix",
        "observed_unix",
        "marker_path",
    ]
    write_csv(destination / "shortfall_scenes.csv", shortfalls, short_fields)
    write_csv(
        destination / "error_scenes.csv",
        errors,
        ["token", "split", "log_name", "category", "high_bank_complete", "detail", "marker_path"],
    )
    payload = "".join(r["token"] + "\n" for r in shortfalls).encode()
    atomic(destination / "shortfall_tokens.txt", lambda f: f.write(payload))
    write_json(
        destination / "report.json", dict(summary=summary, shortfalls=shortfalls, errors=errors)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--report-dir")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    destination = args.report_dir or str(
        Path(args.root)
        / "reports"
        / datetime.now(timezone.utc).strftime("scene_exceptions_%Y%m%dT%H%M%S_%fZ")
    )
    result = collect(args.root, destination, args.workers)
    print(json.dumps(dict(report_directory=destination, summary=result), indent=2), flush=True)


if __name__ == "__main__":
    main()
