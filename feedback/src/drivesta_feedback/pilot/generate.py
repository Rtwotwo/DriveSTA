"""One navtest scene: GT-centred endpoint grid, TTO warp, official PDM labels.

CPU only; does not load or train any learned model. Every trajectory is scored
individually with the unmodified official reference-plus-candidate evaluator.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import lzma
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np
import torch

from drivesta_feedback.paths import bootstrap_navsim

ROOT = bootstrap_navsim()

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.tools.trajectory_mining.path_endpoint_surface import _build_warp
from navsim.evaluate.pdm_score import pdm_score, pdm_score_with_details
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

TOKEN = "dd2d7e23754e58d3"
LOG = "2021.10.06.08.16.17_veh-52_01590_01725"
COMPONENTS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
]


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def init_worker(cache_path):
    global CACHE, SAMPLING, SIMULATOR, SCORER
    torch.set_num_threads(1)
    with lzma.open(cache_path, "rb") as f:
        CACHE = pickle.load(f)
    SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
    SIMULATOR = PDMSimulator(SAMPLING)
    SCORER = PDMScorer(SAMPLING, PDMScorerConfig())


def score_one(item):
    idx, poses = item
    start = time.perf_counter()
    try:
        result, details = pdm_score_with_details(
            CACHE, Trajectory(poses), SAMPLING, SIMULATOR, SCORER
        )
        values = np.array([float(getattr(result, name)) for name in COMPONENTS])
        if not np.isfinite(values).all() or (values < -1e-8).any() or (values > 1 + 1e-8).any():
            raise ValueError(f"Invalid official labels: {values}")
        simulated = np.asarray(details["simulated_states_pred"])
        return idx, values, simulated, time.perf_counter() - start, None
    except Exception:
        return idx, None, None, time.perf_counter() - start, traceback.format_exc()


def make_inputs(cache_path, rx, ry, spacing):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        log_names=[LOG],
        tokens=[TOKEN],
        max_scenes=1,
    )
    loader = SceneLoader(
        ROOT / "data/navsim_logs/test",
        ROOT / "data/sensor_blobs",
        scene_filter,
        SensorConfig.build_no_sensors(),
    )
    assert loader.tokens == [TOKEN], loader.tokens
    scene = loader.get_scene_from_token(TOKEN)
    gt = scene.get_future_trajectory(num_trajectory_frames=8).poses.astype(np.float64)
    assert gt.shape == (8, 3) and np.isfinite(gt).all()
    with lzma.open(cache_path, "rb") as f:
        cache = pickle.load(f)
    current = scene.frames[scene.scene_metadata.num_history_frames - 1]
    scene_pose = np.asarray(current.ego_status.ego_pose)
    cache_pose = np.array(
        [
            cache.ego_state.rear_axle.x,
            cache.ego_state.rear_axle.y,
            cache.ego_state.rear_axle.heading,
        ]
    )
    pose_error = scene_pose - cache_pose
    pose_error[2] = np.arctan2(np.sin(pose_error[2]), np.cos(pose_error[2]))
    assert np.max(np.abs(pose_error)) < 1e-4, pose_error
    assert int(current.timestamp) == cache.ego_state.time_point.time_us
    ix = np.arange(-round(rx / spacing), round(rx / spacing) + 1)
    iy = np.arange(-round(ry / spacing), round(ry / spacing) + 1)
    gx, gy = np.meshgrid(ix, iy, indexing="xy")
    offsets = np.column_stack([gx.ravel(), gy.ravel()]) * spacing
    goals = offsets + gt[-1, :2]
    n = len(goals)
    with torch.no_grad():
        tr, reached, _ = _build_warp(
            torch.from_numpy(gt.astype(np.float32))[None],
            torch.ones(n, 1),
            torch.zeros(n, dtype=torch.long),
            torch.from_numpy(goals.astype(np.float32)),
            torch.ones(n),
            torch.float32,
        )
    trajectories = tr.numpy()
    endpoint_error = np.linalg.norm(trajectories[:, -1, :2] - goals, axis=-1)
    first_xy_error = np.linalg.norm(trajectories[:, 0, :2] - gt[0, :2], axis=-1)
    assert np.isfinite(trajectories).all()
    assert endpoint_error.max() < 3e-5, endpoint_error.max()
    assert first_xy_error.max() < 3e-5
    assert len(np.unique(trajectories[:, -1, :2], axis=0)) == n
    center = int(np.flatnonzero((offsets == 0).all(axis=1))[0])
    heading_diff = trajectories[center, :, 2] - gt[:, 2]
    metadata = dict(
        token=TOKEN,
        log=LOG,
        scene_selection="previously discussed scene; fixed before scoring",
        split="navtest",
        model_training=False,
        axes="current ego frame x forward, y left",
        radius_x_m=rx,
        radius_y_m=ry,
        spacing_m=spacing,
        nx=len(ix),
        ny=len(iy),
        n=n,
        trajectory_num_poses=8,
        trajectory_interval_s=0.5,
        horizon_s=4.0,
        simulator_num_poses=40,
        simulator_interval_s=0.1,
        source_gt="Scene.get_future_trajectory(num_trajectory_frames=8)",
        constructor="project _build_warp; single expert GT reference; factors=1; no goal shrink",
        evaluator="unmodified navsim.evaluate.pdm_score_with_details, reference + one candidate per call",
        score_unit="raw 0..1; PDMS points = 100*raw",
        cache_path=str(cache_path),
        timestamp_us=int(current.timestamp),
        ego_pose=cache_pose.tolist(),
        scene_cache_max_pose_error=float(np.max(np.abs(pose_error))),
        max_planned_endpoint_error_m=float(endpoint_error.max()),
        max_first_future_xy_error_m=float(first_xy_error.max()),
        center_index=center,
        center_vs_original_gt_max_heading_change_rad=float(
            np.abs(np.arctan2(np.sin(heading_diff), np.cos(heading_diff))).max()
        ),
        gt_endpoint=gt[-1].tolist(),
        component_names=COMPONENTS,
        pdm_config=asdict(PDMScorerConfig()),
        environment=dict(
            python=sys.version, torch=torch.__version__, numpy=np.__version__, device="cpu"
        ),
    )
    return gt, trajectories, goals, offsets, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rx", type=float, default=5.0)
    parser.add_argument("--ry", type=float, default=2.0)
    parser.add_argument("--spacing", type=float, default=0.1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cache_path = (
        ROOT / "exp/navsim_cache/metric_cache" / LOG / "unknown" / TOKEN / "metric_cache.pkl"
    )
    gt, traj, goals, offsets, meta = make_inputs(cache_path, args.rx, args.ry, args.spacing)
    meta.update(
        mode=args.mode, workers=args.workers, started_utc=datetime.now(timezone.utc).isoformat()
    )
    write_json(args.output / "config.json", meta)
    np.savez_compressed(
        args.output / "candidates.npz",
        expert_trajectory=gt,
        trajectories=traj,
        requested_goals=goals,
        offsets=offsets,
    )
    print(
        f'GENERATED {len(traj)} endpoint_error={meta["max_planned_endpoint_error_m"]:.3g}',
        flush=True,
    )
    init_worker(cache_path)
    gt_result = score_one((-1, gt))
    if gt_result[-1]:
        raise RuntimeError(gt_result[-1])
    write_json(args.output / "original_gt.json", dict(zip(COMPONENTS, gt_result[1].tolist())))
    np.save(args.output / "original_gt_simulated.npy", gt_result[2])
    n = len(traj)
    if args.mode == "smoke":
        indices = sorted(
            set(
                [
                    0,
                    meta["nx"] - 1,
                    n - meta["nx"],
                    n - 1,
                    meta["center_index"],
                    n // 4,
                    n // 2,
                    3 * n // 4,
                ]
            )
        )
    else:
        indices = list(range(n))
    score_start = time.perf_counter()
    results = []
    if args.mode == "smoke":
        results = [score_one((i, traj[i])) for i in indices]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("fork"),
            initializer=init_worker,
            initargs=(str(cache_path),),
        ) as pool:
            futures = [pool.submit(score_one, (i, traj[i])) for i in indices]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 256 == 0 or len(results) == len(indices):
                    progress = dict(
                        scored=len(results),
                        total=len(indices),
                        errors=sum(r[-1] is not None for r in results),
                        elapsed_seconds=time.perf_counter() - score_start,
                    )
                    write_json(args.output / "progress.json", progress)
                    print("PROGRESS " + json.dumps(progress), flush=True)
    scoring_elapsed = time.perf_counter() - score_start
    results.sort(key=lambda r: r[0])
    failures = [dict(index=r[0], traceback=r[-1]) for r in results if r[-1]]
    write_json(args.output / "errors.json", failures)
    successful = [r for r in results if not r[-1]]
    ids = np.array([r[0] for r in successful], dtype=np.int64)
    scores = np.stack([r[1] for r in successful])
    simulated = np.stack([r[2] for r in successful])
    durations = np.array([r[3] for r in successful])
    np.savez_compressed(
        args.output / "official_scores.npz",
        indices=ids,
        scores=scores,
        simulated_states=simulated,
        per_call_seconds=durations,
        component_names=np.array(COMPONENTS),
    )
    header = "index,offset_x_m,offset_y_m,goal_x_m,goal_y_m," + ",".join(COMPONENTS)
    np.savetxt(
        args.output / "labels.csv",
        np.column_stack([ids, offsets[ids], goals[ids], scores]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )
    # Independent entry-point and call-order check on deterministic probe IDs.
    probes = [
        int(ids[0]),
        int(ids[len(ids) // 2]),
        int(ids[-1]),
        int(ids[np.argmax(scores[:, -1])]),
        int(ids[np.argmin(scores[:, -1])]),
    ]
    parity = []
    for idx in sorted(set(probes), reverse=True):
        plain = pdm_score(CACHE, Trajectory(traj[idx]), SAMPLING, SIMULATOR, SCORER)
        vector = np.array([float(getattr(plain, name)) for name in COMPONENTS])
        previous = scores[int(np.flatnonzero(ids == idx)[0])]
        parity.append(
            dict(index=idx, max_abs_component_difference=float(np.max(np.abs(vector - previous))))
        )
    write_json(args.output / "official_entrypoint_parity.json", parity)
    assert max(p["max_abs_component_difference"] for p in parity) < 1e-9
    values = scores[:, -1] * 100
    summary = dict(
        scene_token=TOKEN,
        mode=args.mode,
        generated=n,
        scored=len(ids),
        failed=len(failures),
        original_gt_pdms=float(gt_result[1][-1] * 100),
        center_warp_pdms=float(
            scores[int(np.flatnonzero(ids == meta["center_index"])[0]), -1] * 100
        ),
        min_pdms=float(values.min()),
        max_pdms=float(values.max()),
        mean_pdms=float(values.mean()),
        median_pdms=float(np.median(values)),
        std_pdms=float(np.std(values)),
        quantile_pdms={str(q): float(np.quantile(values, q)) for q in [0.05, 0.25, 0.75, 0.95]},
        zero_count=int(np.count_nonzero(values == 0)),
        perfect_count=int(np.count_nonzero(values == 100)),
        at_least_90_count=int(np.count_nonzero(values >= 90)),
        strictly_better_than_original_gt=int(
            np.count_nonzero(values > gt_result[1][-1] * 100 + 1e-8)
        ),
        score_values_rounded_6dp=len(np.unique(np.round(values, 6))),
        best_index=int(ids[np.argmax(values)]),
        worst_index=int(ids[np.argmin(values)]),
        best_offset_m=offsets[ids[np.argmax(values)]].tolist(),
        score_elapsed_seconds=scoring_elapsed,
        scoring_save_verify_elapsed_seconds=time.perf_counter() - score_start,
        total_elapsed_seconds=time.perf_counter() - start,
        component_unique_values={
            name: (
                np.unique(scores[:, j]).tolist()
                if len(np.unique(scores[:, j])) < 20
                else dict(
                    min=float(scores[:, j].min()),
                    max=float(scores[:, j].max()),
                    count=len(np.unique(scores[:, j])),
                )
            )
            for j, name in enumerate(COMPONENTS)
        },
        verification=dict(
            max_planned_endpoint_error_m=meta["max_planned_endpoint_error_m"],
            max_official_entrypoint_difference=max(
                p["max_abs_component_difference"] for p in parity
            ),
        ),
        complete=len(failures) == 0 and len(ids) == len(indices),
    )
    write_json(args.output / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} official scoring failures; preserved in errors.json")


if __name__ == "__main__":
    main()
