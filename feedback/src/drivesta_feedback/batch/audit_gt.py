"""Rescore original scene GTs and inspect progress normalization, without bank edits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

from .store import read_json, write_json


def audit_scene(item, source, timestamp_resample=False):
    import torch
    from .source import load_scene, construct, Evaluator, COMPONENTS
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score_with_details

    torch.set_num_threads(1)
    record = item["record"]
    directory = Path(source) / "scenes" / record["split"] / record["token"]
    contract = read_json(directory / "contract.json")
    expert = read_json(directory / "expert.json")
    scene, gt, cache, identity = load_scene(record, [record["token"]], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Scene/cache identity differs from original scene contract")
    if not np.array_equal(gt, np.asarray(expert["trajectory"])):
        raise ValueError("Original scene GT differs from saved expert trajectory")
    start = scene.scene_metadata.num_history_frames - 1
    timestamps = np.array([f.timestamp for f in scene.frames[start : start + 9]], dtype=np.int64)
    intervals = np.diff(timestamps) / 1e6
    zero = construct(gt, np.zeros((64, 2), dtype=np.int32), contract["protocol"])[0][0]
    trajectories = [("raw_gt", gt), ("float32_gt", gt.astype(np.float32)), ("zero_warp", zero)]
    timing_diagnostic = {}
    if timestamp_resample:
        n = len(scene.frames) - start - 1
        original = np.vstack(
            (np.zeros((1, 3)), scene.get_future_trajectory(num_trajectory_frames=n).poses)
        )
        times = (
            np.array([f.timestamp for f in scene.frames[start:]], dtype=np.int64) - timestamps[0]
        ) / 1e6
        desired = np.arange(1, 9, dtype=np.float64) * 0.5
        if np.any(np.diff(times) <= 0) or times[-1] < desired[-1]:
            raise ValueError("Actual scene timestamps cannot support four-second interpolation")
        interpolated = np.column_stack(
            [np.interp(desired, times, original[:, i]) for i in (0, 1)]
            + [np.interp(desired, times, np.unwrap(original[:, 2]))]
        )
        interpolated[:, 2] = np.arctan2(np.sin(interpolated[:, 2]), np.cos(interpolated[:, 2]))
        trajectories.append(("timestamp_resampled_gt", interpolated))
        timing_diagnostic = dict(
            actual_relative_frame_times_s=times.tolist(),
            nominal_output_times_s=desired.tolist(),
            timestamp_resampled_trajectory=interpolated.tolist(),
            method="linear XY and unwrapped heading at actual timestamps",
        )
    arms = {}
    for name, trajectory in trajectories:
        ev = Evaluator(cache)
        values, details = pdm_score_with_details(
            cache, Trajectory(trajectory), ev.sampling, ev.simulator, ev.scorer
        )
        labels = np.array([float(getattr(values, k)) for k in COMPONENTS])
        if not np.isfinite(labels).all():
            raise ValueError("Nonfinite official labels")
        reconstructed = labels[0] * labels[1] * (5 * labels[2] + 5 * labels[3] + 2 * labels[4]) / 12
        if abs(reconstructed - labels[-1]) > 1e-12:
            raise ValueError("PDM score and configured component formula disagree")
        tracker_error = np.linalg.norm(
            details["pred_states"][:, :2] - details["simulated_states_pred"][:, :2], axis=1
        )
        arms[name] = dict(
            labels=labels.tolist(),
            pdms=float(labels[-1] * 100),
            reference_progress_m=float(ev.scorer._progress_raw[0]),
            gt_progress_m=float(ev.scorer._progress_raw[1]),
            reference_multiplicative=float(ev.scorer._multi_metrics[:, 0].prod()),
            mean_tracking_xy_error_m=float(tracker_error.mean()),
            max_tracking_xy_error_m=float(tracker_error.max()),
        )
    raw = np.array(arms["raw_gt"]["labels"])
    return dict(
        token=record["token"],
        split=record["split"],
        log_name=record["log_name"],
        source_cache_path=identity["source_cache_path"],
        official_cache_path=identity["official_cache_path"],
        scene_cache_pose_error=identity["pose_error"],
        frame_intervals_s=intervals.tolist(),
        max_frame_interval_error_s=float(abs(intervals - 0.5).max()),
        original_gt_match=True,
        timing_diagnostic=timing_diagnostic,
        max_stored_label_error=float(abs(raw - expert["labels"]).max()),
        max_cohort_label_error=float(abs(raw - item["gt_components"]).max()),
        arms=arms,
        zero_warp_heading_error_rad=float(
            abs(np.arctan2(np.sin(zero[:, 2] - gt[:, 2]), np.cos(zero[:, 2] - gt[:, 2]))).max()
        ),
    )


def summarize(results):
    labels = np.array([r["arms"]["raw_gt"]["labels"] for r in results])
    pdms = labels[:, -1] * 100
    return dict(
        count=len(results),
        min_pdms=float(pdms.min()),
        mean_pdms=float(pdms.mean()),
        median_pdms=float(np.median(pdms)),
        max_pdms=float(pdms.max()),
        bins={
            "below80": int((pdms < 80).sum()),
            "80to90": int(((pdms >= 80) & (pdms < 90)).sum()),
            "90to95": int(((pdms >= 90) & (pdms < 95)).sum()),
            "95plus": int((pdms >= 95).sum()),
        },
        component_below_one={
            name: int((labels[:, i] < 1).sum())
            for i, name in enumerate(
                [
                    "no_at_fault_collisions",
                    "drivable_area_compliance",
                    "ego_progress",
                    "time_to_collision_within_bound",
                    "comfort",
                    "driving_direction_compliance",
                ]
            )
        },
        only_progress_penalty=int(
            ((labels[:, [0, 1, 3, 4]] == 1).all(axis=1) & (labels[:, 2] < 1)).sum()
        ),
        max_stored_label_error=max(r["max_stored_label_error"] for r in results),
        max_cohort_label_error=max(r["max_cohort_label_error"] for r in results),
        max_scene_cache_pose_error=max(r["scene_cache_pose_error"] for r in results),
        max_frame_interval_error_s=max(r["max_frame_interval_error_s"] for r in results),
        max_float32_pdms_difference=max(
            abs(r["arms"]["float32_gt"]["pdms"] - r["arms"]["raw_gt"]["pdms"]) for r in results
        ),
        zero_warp_pdms_difference_range=[
            min(r["arms"]["zero_warp"]["pdms"] - r["arms"]["raw_gt"]["pdms"] for r in results),
            max(r["arms"]["zero_warp"]["pdms"] - r["arms"]["raw_gt"]["pdms"] for r in results),
        ],
        float32_at_least95=sum(r["arms"]["float32_gt"]["pdms"] >= 95 for r in results),
        zero_warp_at_least95=sum(r["arms"]["zero_warp"]["pdms"] >= 95 for r in results),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--timestamp-resample",
        action="store_true",
        help="Additional diagnostic only; never replace stored GT",
    )
    args = parser.parse_args()
    tokens = args.tokens.read_text().split()
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate input tokens")
    lookup = {x["token"]: x for x in read_json(args.cohort)}
    items = [lookup[t] for t in sorted(tokens)]
    if args.limit:
        items = items[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise ValueError("Audit output is already complete")
    write_json(
        args.output / "invocation.json",
        dict(
            source=str(args.source),
            cohort=str(args.cohort),
            token_file=str(args.tokens),
            tokens=[x["token"] for x in items],
            workers=args.workers,
            timestamp_resample=args.timestamp_resample,
            purpose="Fresh official GT rescore; source banks are read only",
            started_unix=time.time(),
        ),
    )
    results, errors = [], []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        pending = {
            pool.submit(audit_scene, x, str(args.source), args.timestamp_resample): x["token"]
            for x in items
        }
        for f in as_completed(pending):
            try:
                result = f.result()
                results.append(result)
                write_json(args.output / "scenes" / (result["token"] + ".json"), result)
            except Exception:
                error = dict(token=pending[f], traceback=traceback.format_exc())
                errors.append(error)
                print(json.dumps(error), flush=True)
            if (len(results) + len(errors)) % 32 == 0:
                print(
                    json.dumps(
                        dict(
                            processed=len(results),
                            errors=len(errors),
                            elapsed_s=time.time() - start,
                        )
                    ),
                    flush=True,
                )
    results.sort(key=lambda r: r["token"])
    if results:
        summary = summarize(results)
        rows = []
        for r in results:
            raw = r["arms"]["raw_gt"]
            row = {
                k: r[k]
                for k in (
                    "token",
                    "split",
                    "log_name",
                    "max_stored_label_error",
                    "max_frame_interval_error_s",
                )
            }
            row.update({k: v for k, v in raw.items() if k != "labels"})
            row.update(
                dict(
                    zip(
                        ["NC", "DAC", "EP", "TTC", "comfort", "direction", "raw_score"],
                        raw["labels"],
                    )
                )
            )
            row.update(
                float32_pdms=r["arms"]["float32_gt"]["pdms"],
                zero_warp_pdms=r["arms"]["zero_warp"]["pdms"],
            )
            if args.timestamp_resample:
                row.update(
                    timestamp_resampled_pdms=r["arms"]["timestamp_resampled_gt"]["pdms"],
                    timestamp_resampled_comfort=r["arms"]["timestamp_resampled_gt"]["labels"][4],
                )
            rows.append(row)
        with (args.output / "gt_scores.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary = {}
    if args.timestamp_resample and results:
        new = np.array([r["arms"]["timestamp_resampled_gt"]["labels"] for r in results])
        old = np.array([r["arms"]["raw_gt"]["labels"] for r in results])
        summary["timestamp_resample"] = dict(
            above95=int((new[:, -1] >= 0.95).sum()),
            original_comfort_failures=int((old[:, 4] < 1).sum()),
            new_comfort_failures=int((new[:, 4] < 1).sum()),
            mean_pdms=float(new[:, -1].mean() * 100),
            mean_pdms_change=float((new[:, -1] - old[:, -1]).mean() * 100),
            min_pdms=float(new[:, -1].min() * 100),
            max_pdms=float(new[:, -1].max() * 100),
        )
    summary.update(
        requested=len(items),
        errors=errors,
        elapsed_s=time.time() - start,
        completed_unix=time.time(),
    )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("GT audit contains errors")


if __name__ == "__main__":
    main()
