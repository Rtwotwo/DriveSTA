"""Read-only density replay and bounded official-evaluator timing.

Replays thin the measured sampling footprint; they do not rerun adaptive sampling.
Timing uses saved candidates and full official caches, with one candidate per call.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import cProfile
import csv
import io
import json
import lzma
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import pstats
import time

import numpy as np
from scipy.spatial import cKDTree

from .pack import PackedDataset, ROW_DTYPE
from .store import read_json, write_json


def density_mask(keys, phases, names, spacing, boundary_spacing=None):
    """Keep coarse/probes; thin base/dense on a GT-anchored lattice.

    Fine-only boundary samples are excluded when boundary_spacing is None.
    All spacings must be multiples of the measured 0.05 m integer lattice.
    """
    stride = int(round(spacing / 0.05))
    if stride < 1 or abs(stride * 0.05 - spacing) > 1e-8:
        raise ValueError("Spacing must be a multiple of 0.05 m")
    lattice = np.all(keys % stride == 0, axis=1)
    keep = np.zeros(len(keys), dtype=bool)
    for i, name in enumerate(names):
        group = phases == i
        if "coarse" in name or "probes" in name:
            keep |= group
        elif "boundary" in name:
            if boundary_spacing is not None:
                bs = int(round(boundary_spacing / 0.05))
                if bs < 1 or abs(bs * 0.05 - boundary_spacing) > 1e-8:
                    raise ValueError("Invalid boundary spacing")
                keep |= group & np.all(keys % bs == 0, axis=1)
        else:
            keep |= group & lattice
    return keep


def replay(run, output):
    data = PackedDataset(run / "packed", "train")
    policies = [
        ("full", None, None),
        ("dense_010_no005", 0.1, None),
        ("dense_020_no005", 0.2, None),
        ("dense_020_keep005", 0.2, 0.05),
        ("dense_030_no005", 0.3, None),
        ("dense_030_keep005", 0.3, 0.05),
        ("dense_040_no005", 0.4, None),
        ("dense_050_no005", 0.5, None),
    ]
    rows = []
    stage_counts = Counter()
    per_scene_stages = []
    for index in range(len(data)):
        info, samples = data[index]
        full_high = samples["labels"][:, -1] >= 0.95
        for i, name in enumerate(info["phases"]):
            count = int((samples["phase"] == i).sum())
            stage_counts[name] += count
            per_scene_stages.append(dict(token=info["token"], phase=name, count=count))
        for name, spacing, boundary in policies:
            keep = (
                np.ones(len(samples), dtype=bool)
                if spacing is None
                else density_mask(
                    samples["keys"], samples["phase"], info["phases"], spacing, boundary
                )
            )
            selected = samples[keep]
            high = selected["labels"][:, -1] >= 0.95
            distance = np.array([], dtype=float)
            if high.any() and full_high.any():
                distance = cKDTree(selected["keys"][high] * 0.05).query(
                    samples["keys"][full_high] * 0.05
                )[0]
            rows.append(
                dict(
                    token=info["token"],
                    policy=name,
                    count=len(selected),
                    full_count=len(samples),
                    high_count=int(high.sum()),
                    zero_count=int((selected["labels"][:, -1] == 0).sum()),
                    oracle=float(selected["labels"][:, -1].max() * 100),
                    oracle_loss=float(
                        (samples["labels"][:, -1].max() - selected["labels"][:, -1].max()) * 100
                    ),
                    full_high_count=int(full_high.sum()),
                    high_within_015=int((distance <= 0.150001).sum()),
                    high_distance_p95_m=(
                        float(np.quantile(distance, 0.95)) if len(distance) else None
                    ),
                    high_distance_max_m=float(distance.max()) if len(distance) else None,
                )
            )
    summary = []
    for name, _, _ in policies:
        subset = [r for r in rows if r["policy"] == name]
        count = sum(r["count"] for r in subset)
        full = sum(r["full_count"] for r in subset)
        summary.append(
            dict(
                policy=name,
                total_candidates=count,
                mean_candidates=count / len(data),
                reduction_percent=100 * (1 - count / full),
                mean_oracle_loss=float(np.mean([r["oracle_loss"] for r in subset])),
                max_oracle_loss=max(r["oracle_loss"] for r in subset),
                oracle_unchanged_scenes=sum(r["oracle_loss"] < 1e-5 for r in subset),
                high_within_015_fraction=sum(r["high_within_015"] for r in subset)
                / sum(r["full_high_count"] for r in subset),
                full_103288_candidates=count / len(data) * 103288,
                full_103288_packed_gb=count / len(data) * 103288 * ROW_DTYPE.itemsize / 1e9,
            )
        )
    write_json(
        output / "density.json",
        dict(
            scope="fixed measured footprint; no adaptive rerun or scorer training",
            scenes=len(data),
            stage_counts=dict(stage_counts),
            policies=summary,
        ),
    )
    for filename, records in [
        ("density_by_scene.csv", rows),
        ("phases_by_scene.csv", per_scene_stages),
    ]:
        with (output / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary


def score_probe(task):
    run, index, sample_count, repeat = task
    from .source import Evaluator, construct

    started = time.perf_counter()
    data = PackedDataset(Path(run) / "packed", "train")
    info, rows = data[index]
    contract = read_json(Path(run) / "scenes" / "train" / info["token"] / "contract.json")
    with lzma.open(contract["identity"]["official_cache_path"], "rb") as f:
        cache = pickle.load(f)
    rng = np.random.default_rng(1430 + index * 100 + repeat)
    indices = rng.choice(len(rows), size=min(sample_count, len(rows)), replace=False)
    sample = rows[indices]
    evaluator = Evaluator(cache)
    # Warmup kept outside timing; validation uses the stored official labels.
    evaluator.score(sample["trajectories"][0])
    t = time.perf_counter()
    paths, _ = construct(
        np.asarray(contract["identity"]["expert_gt"]), sample["keys"], contract["protocol"]
    )
    construction = time.perf_counter() - t
    construction_difference = float(abs(paths - sample["trajectories"]).max())
    if construction_difference > 3e-5:
        raise ValueError("Reconstruction exceeds tolerance: " + str(construction_difference))
    score_start = time.perf_counter()
    cpu_start = time.process_time()
    # Re-score exact stored paths: float32 batched construction can vary by ULP.
    labels = np.stack([evaluator.score(trajectory) for trajectory in sample["trajectories"]])
    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - score_start
    error = float(abs(labels - sample["labels"].astype(float)).max())
    if error > 1e-6:
        raise ValueError("Profile labels differ from packed labels")
    return dict(
        token=info["token"],
        repeat=repeat,
        count=len(paths),
        score_wall_s=wall,
        score_cpu_s=cpu,
        construction_s=construction,
        total_s=time.perf_counter() - started,
        max_packed_label_error=error,
        construction_max_abs_difference=construction_difference,
        full_count=len(rows),
    )


def benchmark(run, output, workers, samples, repeats):
    from .source import Evaluator  # Warm Python imports before fork; no CUDA.
    import torch

    torch.set_num_threads(1)
    data = PackedDataset(run / "packed", "train")
    tasks = [(str(run), i, samples, rep) for rep in range(repeats) for i in range(len(data))]
    summaries = []
    for count in workers:
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context("fork")) as pool:
            records = list(pool.map(score_probe, tasks))
        elapsed = time.perf_counter() - start
        n = sum(r["count"] for r in records)
        # Scene-weighted extrapolation keeps differing scene sizes and costs paired.
        full_work_cpu = (
            sum(r["full_count"] * r["score_cpu_s"] / r["count"] for r in records) / repeats
        )
        full_work_wall = (
            sum(r["full_count"] * r["score_wall_s"] / r["count"] for r in records) / repeats
        )
        result = dict(
            workers=count,
            jobs=len(records),
            measured_labels=n,
            total_wall_s=elapsed,
            effective_labels_per_s=n / elapsed,
            mean_score_cpu_ms=sum(r["score_cpu_s"] for r in records) / n * 1000,
            mean_score_wall_ms=sum(r["score_wall_s"] for r in records) / n * 1000,
            mean_construct_ms=sum(r["construction_s"] for r in records) / n * 1000,
            estimated_full_label_days_wall=full_work_wall / len(data) * 103288 / count / 86400,
            estimated_full_label_days_cpu_ideal=full_work_cpu / len(data) * 103288 / count / 86400,
            max_label_error=max(r["max_packed_label_error"] for r in records),
        )
        write_json(output / f"timing_{count}.json", dict(summary=result, jobs=records))
        summaries.append(result)
        print(json.dumps(result), flush=True)
    profiler = cProfile.Profile()
    profiler.runcall(score_probe, (str(run), 0, 128, 77))
    profiler.dump_stats(str(output / "official_score.prof"))
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats("cumulative").print_stats(35)
    (output / "hotspots.txt").write_text(buffer.getvalue())
    write_json(
        output / "timing.json",
        dict(
            scope="warm imports, saved official caches; construction and singleton official scoring; excludes scene loading/cache creation, adaptive planning, writes, audit and pack",
            workers=summaries,
        ),
    )
    return summaries


def project_costs(output, policies, timing_directories):
    rows = list(csv.DictReader((output / "density_by_scene.csv").open()))
    evidence = {}
    for directory in timing_directories:
        timing = read_json(directory / "timing.json")
        for summary in timing["workers"]:
            evidence[summary["workers"]] = read_json(
                directory / f"timing_{summary['workers']}.json"
            )
    projections = []
    for workers, record in sorted(evidence.items()):
        timing, jobs = record["summary"], record["jobs"]
        repeats = Counter(j["token"] for j in jobs)
        for policy in policies:
            retained = {
                r["token"]: int(r["count"]) for r in rows if r["policy"] == policy["policy"]
            }
            if set(retained) != set(repeats):
                raise ValueError("Timing scenes must match density scenes")
            seconds = sum(
                retained[j["token"]] * j["score_wall_s"] / j["count"] / repeats[j["token"]]
                for j in jobs
            )
            projections.append(
                dict(
                    policy=policy["policy"],
                    workers=workers,
                    estimated_label_days=seconds / len(retained) * 103288 / workers / 86400,
                    measured_throughput_days=policy["full_103288_candidates"]
                    / timing["effective_labels_per_s"]
                    / 86400,
                )
            )
    write_json(
        output / "projections.json",
        dict(
            scope="short label-throughput extrapolation over 16 scenes; excludes production overhead; density-specific scoring cost assumed to follow the source scene pool",
            timing_sources=[str(p) for p in timing_directories],
            estimates=projections,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, nargs="*", default=[])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--timing-input", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    policies = replay(args.run, args.output)
    print(json.dumps(policies), flush=True)
    if args.workers:
        benchmark(args.run, args.output, args.workers, args.samples, args.repeats)
    timing_sources = args.timing_input + ([args.output] if args.workers else [])
    if timing_sources:
        project_costs(args.output, policies, timing_sources)


if __name__ == "__main__":
    main()
