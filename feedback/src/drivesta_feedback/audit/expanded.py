"""Independent read-only recomputation of one-scene expanded PDM outputs."""

import argparse
import json
from pathlib import Path
import numpy as np


def run(base, result, output):
    old = np.load(base / "candidates.npz")
    old_s = np.load(base / "official_scores.npz")
    new = np.load(result / "candidates.npz")
    new_s = np.load(result / "official_scores.npz")
    coverage = np.load(result / "coverage_grids.npz")
    summary = json.loads((result / "summary.json").read_text(encoding="utf8"))
    config = json.loads((result / "config.json").read_text(encoding="utf8"))
    keys = new["integer_keys"]
    scores = new_s["scores"]
    old_trajectories, new_trajectories = old["trajectories"], new["trajectories"]
    old_values, old_simulated = old_s["scores"], old_s["simulated_states"]
    new_simulated = new_s["simulated_states"]
    lookup = {tuple(k): i for i, k in enumerate(keys)}
    assert len(lookup) == len(keys)
    assert np.array_equal(new_s["indices"], np.arange(len(keys)))
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
    assert np.isfinite(new["trajectories"]).all()
    assert np.array_equal(old["expert_trajectory"], new["expert_trajectory"])
    assert np.array_equal(keys * 0.05, new["offsets"])
    assert np.array_equal(new["requested_goals"], new["offsets"] + new["expert_trajectory"][-1, :2])
    oldkeys = np.rint(old["offsets"] / 0.05).astype(int)
    for r, index in enumerate(old_s["indices"]):
        i = lookup[tuple(oldkeys[index])]
        assert np.array_equal(old_trajectories[index], new_trajectories[i])
        assert np.array_equal(old_values[r], scores[i])
        assert np.array_equal(old_simulated[r], new_simulated[i])
    fx, fy = coverage["fine_x_keys"], coverage["fine_y_keys"]
    grid = np.full((len(fy), len(fx)), np.nan)
    for j, y in enumerate(fy):
        for i, x in enumerate(fx):
            if (x, y) in lookup:
                grid[j, i] = 100 * scores[lookup[(x, y)], -1]
    assert np.array_equal(grid, coverage["fine_pdms"], equal_nan=True)
    assert np.isfinite(grid[coverage["requested_fine_mask"]]).all()
    for k, val in zip(coverage["coarse_keys"], coverage["coarse_pdms"].ravel()):
        assert val == 100 * scores[lookup[tuple(k)], -1]
    vals = scores[:, -1] * 100
    best = int(np.argmax(vals))
    high = vals >= 95
    fine_high = np.isfinite(grid) & (grid >= 95)
    xx, yy = np.meshgrid(fx * 0.05, fy * 0.05)
    in_old = (xx >= -5) & (xx <= 5) & (yy >= -2) & (yy <= 2)
    checks = dict(
        unique_total=len(keys),
        reused_original=len(oldkeys),
        newly_scored=len(keys) - len(oldkeys),
        max_pdms=float(vals.max()),
        best_index=best,
        fine_measured_nodes=int(np.isfinite(grid).sum()),
        fine_high_nodes=int(fine_high.sum()),
        fine_high_nodes_inside_old=int((fine_high & in_old).sum()),
        fine_high_nodes_outside_old=int((fine_high & ~in_old).sum()),
        mixed_sample_ge95=int(high.sum()),
        mixed_sample_zero=int((vals == 0).sum()),
        mixed_sample_perfect=int(np.isclose(vals, 100, atol=1e-8, rtol=0).sum()),
    )
    for key, value in checks.items():
        assert summary[key] == value, (key, summary[key], value)
    component_rows = {}
    for j, name in enumerate(new_s["component_names"]):
        if j not in (2, 6):
            unique, counts = np.unique(scores[high, j], return_counts=True)
            component_rows[str(name)] = {str(float(v)): int(n) for v, n in zip(unique, counts)}
    edge_nodes = {
        "rear": keys[:, 0] * 0.05 == config["final_bounds_m"][0],
        "front": keys[:, 0] * 0.05 == config["final_bounds_m"][1],
        "right": keys[:, 1] * 0.05 == config["final_bounds_m"][2],
        "left": keys[:, 1] * 0.05 == config["final_bounds_m"][3],
    }
    edges = {
        side: {
            "sampled": int(m.sum()),
            "high": int((m & high).sum()),
            "max_pdms": float(vals[m].max()) if m.any() else None,
        }
        for side, m in edge_nodes.items()
    }
    probe_indices = [lookup[tuple(k)] for k in coverage["probe_keys"]]
    probes_high = high[probe_indices]
    endpoint_error = np.linalg.norm(new["trajectories"][:, -1, :2] - new["requested_goals"], axis=1)
    motion = np.load(result / "motion_diagnostics.npz")
    unknown = ~np.isfinite(grid)
    adjacent_unknown = np.zeros(grid.shape, dtype=bool)
    adjacent_unknown[:-1, :] |= unknown[1:, :]
    adjacent_unknown[1:, :] |= unknown[:-1, :]
    adjacent_unknown[:, :-1] |= unknown[:, 1:]
    adjacent_unknown[:, 1:] |= unknown[:, :-1]
    high_next_to_unknown = fine_high & adjacent_unknown
    high_next_to_unknown_offsets = np.column_stack(
        [xx[high_next_to_unknown], yy[high_next_to_unknown]]
    )
    ddc_valid = scores[:, 5] == 1
    best_ddc = int(np.flatnonzero(ddc_valid)[np.argmax(vals[ddc_valid])])
    perfect_ddc = ddc_valid & np.isclose(vals, 100, atol=1e-8, rtol=0)
    perfect_ddc_offsets = new["offsets"][perfect_ddc]
    report = dict(
        passed=True,
        checks=checks,
        best_offset_m=new["offsets"][best].tolist(),
        old_best_pdms=float(old_s["scores"][:, -1].max() * 100),
        expanded_best_components=dict(
            zip(new_s["component_names"].tolist(), scores[best].tolist())
        ),
        high_offset_min_m=new["offsets"][high].min(axis=0).tolist(),
        high_offset_max_m=new["offsets"][high].max(axis=0).tolist(),
        high_components_sample_counts=component_rows,
        outer_edge_counts_all_measured=edges,
        interior_fine_high_next_to_unknown=dict(
            count=int(high_next_to_unknown.sum()),
            offset_min_m=(
                high_next_to_unknown_offsets.min(axis=0).tolist()
                if len(high_next_to_unknown_offsets)
                else None
            ),
            offset_max_m=(
                high_next_to_unknown_offsets.max(axis=0).tolist()
                if len(high_next_to_unknown_offsets)
                else None
            ),
        ),
        direction_compliant_best=dict(
            index=best_ddc,
            pdms=float(vals[best_ddc]),
            offset_m=new["offsets"][best_ddc].tolist(),
            components=dict(zip(new_s["component_names"].tolist(), scores[best_ddc].tolist())),
        ),
        perfect_and_direction_compliant=dict(
            count=int(perfect_ddc.sum()),
            offset_min_m=(
                perfect_ddc_offsets.min(axis=0).tolist() if len(perfect_ddc_offsets) else None
            ),
            offset_max_m=(
                perfect_ddc_offsets.max(axis=0).tolist() if len(perfect_ddc_offsets) else None
            ),
        ),
        offset_probes=dict(
            total=len(probe_indices),
            high=int(probes_high.sum()),
            max_pdms=float(vals[probe_indices].max()),
        ),
        max_planned_endpoint_error_m=float(endpoint_error.max()),
        high_simulated_endpoint_error_m={
            str(q): float(np.quantile(motion["endpoint_error_m"][high], q))
            for q in (0.5, 0.95, 1.0)
        },
        high_simulated_max_speed_mps={
            str(q): float(np.quantile(motion["max_simulated_speed_mps"][high], q))
            for q in (0.5, 0.95, 1.0)
        },
        note="Sample counts from a mixed-resolution set are not area fractions. Fine-grid counts use common 0.1 m coordinates. Unmeasured cells remain NaN.",
    )
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.result, args.output)


if __name__ == "__main__":
    main()
