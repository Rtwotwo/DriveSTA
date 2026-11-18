"""Compact progress and final validation report, without loading trajectory arrays."""

import argparse
import csv
import io
import time
from pathlib import Path
import numpy as np
from .store import read_json, write_json, atomic


def summarize(run):
    run = Path(run)
    contract = read_json(run / "run.json")
    states = [read_json(p) for p in run.glob("scenes/*/*/progress.json")]
    completed = [read_json(p) for p in sorted(run.glob("scenes/*/*/COMPLETE.json"))]
    for scene in completed:
        directory = run / "scenes" / scene["split"] / scene["token"]
        blocks = list(directory.glob("phases/*/block_*.npz"))
        scene["score_span_s"] = (
            max(p.stat().st_mtime for p in blocks) - (directory / "expert.json").stat().st_mtime
        )
    report = dict(
        planned_scenes=contract["scene_count"],
        started_scenes=len(states),
        complete_scenes=len(completed),
        committed_candidates=sum(s.get("candidates", 0) for s in states),
        error_records=len(list((run / "errors").glob("*.json"))),
        phase_counts={
            phase: sum(s.get("phase", s["status"]) == phase for s in states)
            for phase in sorted(set(s.get("phase", s["status"]) for s in states))
        },
        updated_unix=time.time(),
    )
    if (run / "audit.json").exists():
        audit = read_json(run / "audit.json")
        report["audit_passed"] = audit["passed"]
        if audit["passed"]:
            report["official_checks"] = sum(a["official_checks"] for a in audit["scenes"])
            report["max_official_error"] = max(a["max_official_error"] for a in audit["scenes"])
    if (run / "packed/COMPLETE.json").exists():
        report["packed"] = read_json(run / "packed/COMPLETE.json")
    if len(completed) == contract["scene_count"]:
        report.update(
            candidates=sum(s["candidates"] for s in completed),
            zero_count=sum(s["zero_count"] for s in completed),
            high_count=sum(s["high_count"] for s in completed),
            reused_blocks=sum(s["reused_blocks"] for s in completed),
            max_endpoint_error_m=max(s["max_endpoint_error_m"] for s in completed),
            candidate_cap_scenes=sum(s["candidate_cap_reached"] for s in completed),
            outer_open_scenes=sum(any(s["outer_high_counts"]) for s in completed),
            no_high_scenes=sum(s["high_count"] == 0 for s in completed),
            sampler_fallback_scenes=sum(s["dense_seed_info"]["fallback_used"] for s in completed),
        )
        for field in ("candidates", "score_span_s", "elapsed_s", "max_rss_mib", "bytes"):
            a = np.array([s[field] for s in completed])
            report[field + "_distribution"] = {
                q: float(np.quantile(a, v))
                for q, v in (("min", 0), ("p50", 0.5), ("p95", 0.95), ("max", 1))
            }
        fields = [
            "token",
            "log_name",
            "split",
            "candidates",
            "original_gt_pdms",
            "best_pdms",
            "zero_count",
            "high_count",
            "coverage",
            "candidate_cap_reached",
            "max_endpoint_error_m",
            "reused_blocks",
            "score_span_s",
            "elapsed_s",
            "max_rss_mib",
            "bytes",
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)
        payload = buffer.getvalue().encode()
        atomic(run / "scene_results.csv", lambda f: f.write(payload))
    write_json(run / "status.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    import json

    print(json.dumps(summarize(args.run), indent=2))


if __name__ == "__main__":
    main()
