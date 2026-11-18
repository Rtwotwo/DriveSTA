"""Replay saved GT-neighborhood labels and gates with the source evaluator."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np


def verify(job):
    root, token = job
    root = Path(root)
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    from drivesta_feedback.batch.source import load_scene, Evaluator
    from drivesta_feedback.batch.filter_targets import (
        TargetPolicy,
        preliminary_rules,
        planned_space_checker,
        tracking_errors,
    )
    from drivesta_feedback.batch.recovery_gt_local import perturb_expert
    from drivesta_feedback.batch.recovery_seeds import resample_recorded

    config = json.loads((root / "config.json").read_text())
    row = json.loads((root / "scenes" / (token + ".json")).read_text())
    record = row["record"]
    contract = json.loads(
        (Path(config["source"]) / "scenes" / record["split"] / token / "contract.json").read_text()
    )
    _, expert, cache, identity = load_scene(record, [token], contract["protocol"])
    assert identity == contract["identity"]
    policy = TargetPolicy(**config["policy"])
    ev = Evaluator(cache)
    space = planned_space_checker(cache)
    dyn = cache.ego_state.dynamic_car_state
    v = [dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y]
    a = [dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y]
    p = cache.ego_state.rear_axle
    origin = [p.x, p.y, p.heading]
    limits = np.array(
        [
            policy.max_tracking_ade_m,
            policy.max_tracking_error_m,
            policy.max_tracking_heading_error_rad,
        ]
    )
    with np.load(root / "scenes" / (token + ".npz"), allow_pickle=False) as z:
        t, y, names = z["trajectories"], z["labels"], z["names"]
        assert len(t) == row["queries"] and t.shape == (len(t), 8, 3) and y.shape == (len(t), 7)
        assert np.isfinite(t).all() and np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
        assert len({q.tobytes() for q in t}) == len(t)
        np.testing.assert_allclose(
            y[:, -1],
            y[:, 0] * y[:, 1] * (5 * y[:, 2] + 5 * y[:, 3] + 2 * y[:, 4]) / 12,
            atol=1e-9,
            rtol=0,
        )
        ids = sorted(
            set(np.linspace(0, len(t) - 1, 4).astype(int).tolist() + [int(np.argmax(y[:, -1]))])
        )
        timed = resample_recorded(z["recorded_poses"], z["recorded_times"])
        worst = 0.0
        for i in ids:
            recreated = (
                np.asarray(expert, np.float32)
                if str(names[i]) == "raw_expert"
                else perturb_expert(z["recorded_poses"], z["recorded_times"], z["coefficients"][i])
            )
            np.testing.assert_array_equal(recreated, t[i])
            label, tracked = ev.score(t[i], details=True)
            worst = max(worst, float(abs(label - y[i]).max()))
            masks = preliminary_rules(t[i : i + 1], label[None], expert, policy, v, a)
            failed = {k for k, m in masks.items() if not m[0]}
            if not space(t[i]):
                failed.add("planned_space")
            if np.any(np.array(tracking_errors(t[i], tracked, origin)) > limits):
                failed.add("tracking")
            saved = row["arms"][str(names[i])]
            if config.get("max_gt_deviation_m"):
                deviation = float(np.linalg.norm(t[i, :, :2] - timed[:, :2], axis=1).max())
                assert abs(deviation - saved["max_gt_deviation_m"]) < 1e-6
                if deviation > config["max_gt_deviation_m"]:
                    failed.add("gt_neighborhood_bound")
            assert failed == set(saved["failed_rules"]), (token, i, failed, saved["failed_rules"])
            assert (not failed) == saved["generator_eligible"]
        assert worst <= 1e-9
    return dict(
        run=root.name,
        token=token,
        queries=len(t),
        replayed=len(ids),
        maximum_label_error=worst,
        gate_replay_identical=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    jobs = []
    for root in a.runs:
        cfg = json.loads((root / "config.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        assert not summary["errors"] and summary["scenes"] == len(cfg["tokens"])
        jobs.extend((str(root), t) for t in cfg["tokens"])
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(verify, jobs))
    result = dict(
        scene_runs=len(rows),
        queries=sum(x["queries"] for x in rows),
        official_replays=sum(x["replayed"] for x in rows),
        maximum_label_error=max(x["maximum_label_error"] for x in rows),
        all_gate_replays_identical=True,
        results=rows,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
