"""One-scene official-label domain expansion; no learned model or training."""

import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from pathlib import Path
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label
from drivesta_feedback.pilot import generate as base

UNIT = 0.05


def key_grid(bounds, step):
    lo, hi, low, high = np.rint(np.array(bounds) / UNIT).astype(int)
    stride = int(round(step / UNIT))
    x = np.arange(lo, hi + 1, stride)
    y = np.arange(low, high + 1, stride)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()]), x, y


def construct(gt, keys):
    offsets = np.asarray(keys, dtype=np.int64) * UNIT
    goals = gt[-1, :2] + offsets
    n = len(keys)
    with torch.no_grad():
        tr, _, _ = base._build_warp(
            torch.from_numpy(gt.astype(np.float32))[None],
            torch.ones(n, 1),
            torch.zeros(n, dtype=torch.long),
            torch.from_numpy(goals.astype(np.float32)),
            torch.ones(n),
            torch.float32,
        )
    tr = tr.numpy()
    assert np.isfinite(tr).all()
    assert np.max(np.linalg.norm(tr[:, -1, :2] - goals, axis=1)) < 3e-5
    return tr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    old = np.load(args.base / "candidates.npz")
    old_scores = np.load(args.base / "official_scores.npz")
    old_config = json.loads((args.base / "config.json").read_text())
    gt = old["expert_trajectory"]
    old_keys = np.rint(old["offsets"] / UNIT).astype(int)
    # Reconstruct all old candidates before reusing any label.
    rebuilt = construct(gt, old_keys)
    assert np.array_equal(rebuilt, old["trajectories"]), "Reused construction differs"
    cache_path = old_config["cache_path"]
    checked_gt, _, _, _, current = base.make_inputs(Path(cache_path), 5.0, 2.0, 0.1)
    assert np.array_equal(checked_gt, gt)
    assert current["pdm_config"] == old_config["pdm_config"]
    base.init_worker(cache_path)
    assert base.SCORER._config == base.PDMScorerConfig()
    gt_row = base.score_one((-1, gt))
    if gt_row[-1]:
        raise RuntimeError(gt_row[-1])
    previous_gt = json.loads((args.base / "original_gt.json").read_text())
    assert np.max(np.abs(gt_row[1] - [previous_gt[k] for k in base.COMPONENTS])) < 1e-9
    reuse_parity = []
    for i in sorted(
        set(
            [
                len(old_keys) // 2,
                int(np.argmax(old_scores["scores"][:, -1])),
                int(np.argmin(old_scores["scores"][:, -1])),
            ]
        )
    ):
        saved_idx = int(old_scores["indices"][i])
        check = base.score_one((saved_idx, rebuilt[saved_idx]))
        if check[-1]:
            raise RuntimeError(check[-1])
        err = float(np.max(np.abs(check[1] - old_scores["scores"][i])))
        assert err < 1e-9
        reuse_parity.append(dict(index=saved_idx, max_component_difference=err))
    base.write_json(args.output / "reuse_parity.json", reuse_parity)
    config = dict(old_config)
    config.update(
        mode="expanded_multiresolution",
        started_utc=datetime.now(timezone.utc).isoformat(),
        workers=args.workers,
        initial_bounds_m=[-10, 20, -8, 8],
        hard_bounds_m=[-15, 40, -12, 12],
        expansion_step_m=[5, 2],
        max_expansion_rounds=4,
        high_threshold_pdms=95.0,
        coarse_spacing_m=0.5,
        fine_spacing_m=0.1,
        edge_spacing_m=0.05,
        high_buffer_m=1.1,
        transition_buffer_m=0.5,
        max_unique_candidates=80000,
        uniform_probe="quarter-offset .5m grid at +.25m in both axes",
        selection="official labels only; no learned scorer",
        base_directory=str(args.base),
        motion_scope="GT geometric warp; raw labels kept; no new dynamics rejection",
        reused_trajectory_bitwise_equal=True,
    )
    for stale in ("nx", "ny", "n", "spacing_m", "radius_x_m", "radius_y_m", "center_index"):
        config.pop(stale, None)
    base.write_json(args.output / "config.json", config)
    base.write_json(args.output / "original_gt.json", previous_gt)
    rows = {}
    order = []
    stage_notes = []
    time_sum = 0.0
    for index, values, sim in zip(
        old_scores["indices"], old_scores["scores"], old_scores["simulated_states"]
    ):
        k = tuple(old_keys[index])
        order.append(k)
        rows[k] = (old["trajectories"][index], values, sim, "original", 0.0)
    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("fork"),
        initializer=base.init_worker,
        initargs=(cache_path,),
    )

    def score_keys(keys, stage):
        nonlocal time_sum
        keys = np.unique(np.asarray(keys, dtype=np.int64).reshape(-1, 2), axis=0)
        missing = [tuple(k) for k in keys if tuple(k) not in rows]
        if len(rows) + len(missing) > config["max_unique_candidates"]:
            raise RuntimeError("Predeclared 80000-candidate cap exceeded; preserved prior stages")
        begin = time.perf_counter()
        results = []
        print(
            "START "
            + json.dumps(
                dict(
                    stage=stage,
                    requested=len(keys),
                    new=len(missing),
                    reused=len(keys) - len(missing),
                )
            ),
            flush=True,
        )
        if missing:
            trajectories = construct(gt, missing)
            futures = [pool.submit(base.score_one, (i, t)) for i, t in enumerate(trajectories)]
            for fut in as_completed(futures):
                row = fut.result()
                if row[-1]:
                    base.write_json(
                        args.output / f"{stage}_error.json",
                        dict(key=missing[row[0]], traceback=row[-1]),
                    )
                    raise RuntimeError(row[-1])
                results.append(row)
                if len(results) % 1024 == 0 or len(results) == len(missing):
                    note = dict(
                        stage=stage,
                        completed=len(results),
                        new_total=len(missing),
                        elapsed_s=time.perf_counter() - begin,
                    )
                    base.write_json(args.output / "progress.json", note)
                    print("PROGRESS " + json.dumps(note), flush=True)
            results.sort(key=lambda r: r[0])
            sims = np.stack([r[2] for r in results])
            vals = np.stack([r[1] for r in results])
            np.savez_compressed(
                args.output / f"stage_{stage}.npz",
                keys=np.array(missing),
                trajectories=trajectories,
                scores=vals,
                simulated_states=sims,
            )
            for i, value, sim, seconds, _ in results:
                k = missing[i]
                order.append(k)
                rows[k] = (trajectories[i], value, sim, stage, seconds)
        elapsed = time.perf_counter() - begin
        time_sum += elapsed
        s = np.array([rows[tuple(k)][1][-1] * 100 for k in keys])
        note = dict(
            stage=stage,
            requested=int(len(keys)),
            new=len(missing),
            reused=len(keys) - len(missing),
            max_pdms=float(s.max()),
            ge95=int((s >= 95).sum()),
            zero=int((s == 0).sum()),
            unique_cumulative=len(rows),
            score_save_seconds=elapsed,
        )
        stage_notes.append(note)
        base.write_json(args.output / "stages.json", stage_notes)
        print("STAGE " + json.dumps(note), flush=True)
        return s

    def coarse_field(bounds, stage):
        keys, x, y = key_grid(bounds, 0.5)
        score_keys(keys, stage)
        # score_keys sorts by x then y; explicitly restore the mesh (y then x).
        scores = np.array([rows[tuple(k)][1][-1] * 100 for k in keys]).reshape(len(y), len(x))
        comp = np.array([rows[tuple(k)][1] for k in keys]).reshape(len(y), len(x), 7)
        counts = dict(
            rear=int((scores[:, 0] >= 95).sum()),
            front=int((scores[:, -1] >= 95).sum()),
            right=int((scores[0, :] >= 95).sum()),
            left=int((scores[-1, :] >= 95).sum()),
        )
        np.savez_compressed(
            args.output / f"coarse_{stage}.npz", keys=keys, x=x, y=y, scores=scores, components=comp
        )
        return keys, x, y, scores, comp, counts

    bounds = list(config["initial_bounds_m"])
    expansion = []
    for r in range(config["max_expansion_rounds"] + 1):
        ck, cx, cy, cs, cc, edges = coarse_field(bounds, f"coarse{r}")
        expansion.append(dict(round=r, bounds_m=list(bounds), edge_high_counts=edges))
        base.write_json(args.output / "expansion.json", expansion)
        target = list(bounds)
        if edges["rear"]:
            target[0] = max(-15, bounds[0] - 5)
        if edges["front"]:
            target[1] = min(40, bounds[1] + 5)
        if edges["right"]:
            target[2] = max(-12, bounds[2] - 2)
        if edges["left"]:
            target[3] = min(12, bounds[3] + 2)
        if target == bounds or r == config["max_expansion_rounds"]:
            break
        bounds = target

    # Broad off-grid probes check gaps in the coarse lattice before fine-region construction.
    probes = ck + 5
    inside = (probes[:, 0] <= bounds[1] / UNIT) & (probes[:, 1] <= bounds[3] / UNIT)
    probes = probes[inside]
    probe_scores = score_keys(probes, "offset_probes")
    fine_keys, fx, fy = key_grid(bounds, 0.1)
    fine_shape = (len(fy), len(fx))
    high_seed = np.zeros(fine_shape, bool)
    transition_seed = np.zeros(fine_shape, bool)

    def seed(keys, mask):
        for k in keys:
            ix = int(round((k[0] - fx[0]) / 2))
            iy = int(round((k[1] - fy[0]) / 2))
            if 0 <= ix < len(fx) and 0 <= iy < len(fy):
                mask[iy, ix] = True

    high = [k for k in rows if rows[k][1][-1] * 100 >= 95]
    seed(high, high_seed)
    transition = np.zeros(cs.shape, bool)
    for axis in (0, 1):
        left = np.take(cc, np.arange(cc.shape[axis] - 1), axis=axis)
        right = np.take(cc, np.arange(1, cc.shape[axis]), axis=axis)
        bad = (np.max(np.abs(left - right), axis=-1) > 0.05) | (
            (left[:, :, -1] >= 0.95) != (right[:, :, -1] >= 0.95)
        )
        if axis == 0:
            transition[:-1] |= bad
            transition[1:] |= bad
        else:
            transition[:, :-1] |= bad
            transition[:, 1:] |= bad
    seed(ck[transition.ravel()], transition_seed)
    hdist = distance_transform_edt(~high_seed) * 0.1
    tdist = distance_transform_edt(~transition_seed) * 0.1
    mask = (hdist <= 1.100001) | (tdist <= 0.500001)
    # Retain the original densely evaluated rectangle for a direct before/after view.
    gx, gy = np.meshgrid(fx * UNIT, fy * UNIT)
    mask |= (gx >= -5) & (gx <= 5) & (gy >= -2) & (gy <= 2)
    fine_selected = fine_keys[mask.ravel()]
    score_keys(fine_selected, "fine01")
    grid = np.full(fine_shape, np.nan)
    for i, k in enumerate(fine_keys):
        if tuple(k) in rows:
            grid.flat[i] = rows[tuple(k)][1][-1] * 100
    # 0.05 m cells only where adjacent measured 0.1 m corners straddle the high threshold.
    corners = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:]])
    boundary = (
        np.isfinite(corners).all(axis=0)
        & (np.min(corners, axis=0) < 95)
        & (np.max(corners, axis=0) >= 95)
    )
    iy, ix = np.nonzero(boundary)
    boundary_keys = []
    for y, x in zip(iy, ix):
        boundary_keys.extend(
            (int(fx[x] + dx), int(fy[y] + dy)) for dx in (0, 1, 2) for dy in (0, 1, 2)
        )
    if boundary_keys:
        score_keys(boundary_keys, "boundary005")
    pool.shutdown()
    # Final saved arrays are keyed once per unique endpoint, including all old labels.
    keys = np.array(order, dtype=np.int64)
    trajectories = np.stack([rows[k][0] for k in order])
    scores = np.stack([rows[k][1] for k in order])
    sims = np.stack([rows[k][2] for k in order])
    offsets = keys * UNIT
    goals = offsets + gt[-1, :2]
    assert len(np.unique(keys, axis=0)) == len(keys)
    assert np.isfinite(scores).all() and np.isfinite(sims).all()
    np.savez_compressed(
        args.output / "candidates.npz",
        expert_trajectory=gt,
        trajectories=trajectories,
        requested_goals=goals,
        offsets=offsets,
        integer_keys=keys,
        source_stage=np.array([rows[k][3] for k in order]),
    )
    np.savez_compressed(
        args.output / "official_scores.npz",
        indices=np.arange(len(keys)),
        scores=scores,
        simulated_states=sims,
        component_names=np.array(base.COMPONENTS),
        per_call_seconds=np.array([rows[k][4] for k in order]),
    )
    np.savez_compressed(
        args.output / "coverage_grids.npz",
        fine_x_keys=fx,
        fine_y_keys=fy,
        requested_fine_mask=mask,
        fine_pdms=grid,
        coarse_keys=ck,
        coarse_pdms=cs,
        coarse_components=cc,
        probe_keys=probes,
        boundary_cell_mask=boundary,
    )
    np.savetxt(
        args.output / "labels.csv",
        np.column_stack([np.arange(len(keys)), offsets, goals, scores]),
        delimiter=",",
        header="index,offset_x_m,offset_y_m,goal_x_m,goal_y_m," + ",".join(base.COMPONENTS),
        comments="",
        fmt="%.10g",
    )
    pdms = scores[:, -1] * 100
    best = int(np.argmax(pdms))
    probes_final = sorted(set([0, len(keys) - 1, best, len(keys) // 2]), reverse=True)
    parity = []
    for idx in probes_final:
        res = base.pdm_score(
            base.CACHE,
            base.Trajectory(trajectories[idx]),
            base.SAMPLING,
            base.SIMULATOR,
            base.SCORER,
        )
        vals = np.array([getattr(res, name) for name in base.COMPONENTS])
        err = float(np.max(np.abs(vals - scores[idx])))
        assert err < 1e-9
        parity.append(dict(index=idx, max_abs_difference=err))
    base.write_json(args.output / "official_entrypoint_parity.json", parity)
    pose = np.array(config["ego_pose"])
    delta = sims[:, -1, :2] - pose[:2]
    a = pose[2]
    local = delta @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    end_error = np.linalg.norm(local - goals, axis=1)
    speed = np.linalg.norm(sims[:, :, 3:5], axis=-1).max(axis=1)
    planned = np.concatenate([np.zeros((len(trajectories), 1, 2)), trajectories[:, :, :2]], axis=1)
    velocity = np.diff(planned, axis=1) / 0.5
    planned_speed = np.linalg.norm(velocity, axis=-1).max(axis=1)
    planned_accel = np.linalg.norm(np.diff(velocity, axis=1) / 0.5, axis=-1).max(axis=1)
    np.savez_compressed(
        args.output / "motion_diagnostics.npz",
        simulated_endpoint_ego=local,
        endpoint_error_m=end_error,
        max_simulated_speed_mps=speed,
        max_planned_segment_speed_mps=planned_speed,
        max_planned_internal_accel_mps2=planned_accel,
    )
    # Counts use the common 0.1 m grid only; mixed-density counts are stored separately.
    grid_high = np.isfinite(grid) & (grid >= 95)
    old_in_grid = (gx >= -5) & (gx <= 5) & (gy >= -2) & (gy <= 2)
    n_components = label(grid_high)[1]
    components_high = {
        name: {
            str(v): int(n) for v, n in zip(*np.unique(scores[pdms >= 95, j], return_counts=True))
        }
        for j, name in enumerate(base.COMPONENTS)
        if j not in (2, 6)
    }
    summary = dict(
        scene_token=config["token"],
        complete=True,
        failed=0,
        unique_total=len(keys),
        reused_original=len(old_keys),
        newly_scored=len(keys) - len(old_keys),
        original_gt_pdms=previous_gt["score"] * 100,
        old_max_pdms=float(old_scores["scores"][:, -1].max() * 100),
        max_pdms=float(pdms.max()),
        best_index=best,
        best_offset_m=offsets[best].tolist(),
        best_components=dict(zip(base.COMPONENTS, scores[best].tolist())),
        final_bounds_m=bounds,
        edge_high_counts_coarse=edges,
        edge_high_counts_fine=dict(
            rear=int(grid_high[:, 0].sum()),
            front=int(grid_high[:, -1].sum()),
            right=int(grid_high[0, :].sum()),
            left=int(grid_high[-1, :].sum()),
        ),
        high_region_closed_on_coarse_outer_boundary=not any(edges.values()),
        fine_domain_total_nodes=grid.size,
        fine_requested_nodes=int(mask.sum()),
        fine_requested_nodes_scored=int(np.isfinite(grid[mask]).sum()),
        fine_measured_nodes=int(np.isfinite(grid).sum()),
        fine_high_nodes=int(grid_high.sum()),
        fine_high_nodes_inside_old=int((grid_high & old_in_grid).sum()),
        fine_high_nodes_outside_old=int((grid_high & ~old_in_grid).sum()),
        fine_high_connected_components=int(n_components),
        mixed_sample_ge95=int((pdms >= 95).sum()),
        mixed_sample_zero=int((pdms == 0).sum()),
        mixed_sample_perfect=int(np.isclose(pdms, 100, atol=1e-8, rtol=0).sum()),
        high_components=components_high,
        endpoint_error_all={str(q): float(np.quantile(end_error, q)) for q in (0.5, 0.95, 1.0)},
        endpoint_error_high={
            str(q): float(np.quantile(end_error[pdms >= 95], q)) for q in (0.5, 0.95, 1.0)
        },
        max_planned_endpoint_error_m=float(
            np.linalg.norm(trajectories[:, -1, :2] - goals, axis=1).max()
        ),
        max_parity_error=max(p["max_abs_difference"] for p in parity),
        score_save_seconds=time_sum,
        total_seconds=time.perf_counter() - start,
        notes="Mixed-density counts are sample counts, not area estimates. Outer boundary checks use .5m lattice; finite-resolution evidence.",
    )
    base.write_json(args.output / "summary.json", summary)
    config["final_bounds_m"] = bounds
    base.write_json(args.output / "config.json", config)
    base.write_json(args.output / "errors.json", [])
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
