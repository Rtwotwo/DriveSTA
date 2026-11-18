"""Hide recorded tail, forecast it, and measure geometry / full-v1 label errors."""

import argparse
import copy
import json
import lzma
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def geometry_error(source, forecast, cutoff, end):
    dt = source._sample_interval
    errors = []
    last = []
    missing = 0
    expected = 0
    for i in range(round(cutoff / dt) + 1, round(end / dt) + 1):
        for token in source[i].tokens:
            expected += 1
            if token not in forecast[i].token_to_idx:
                missing += 1
                continue
            a, b = source[i][token].centroid, forecast[i][token].centroid
            err = float(a.distance(b))
            errors.append(err)
            if i == round(end / dt):
                last.append(err)
    return dict(
        distances=errors,
        final_distances=last,
        missing_actor_steps=missing,
        recorded_actor_steps=expected,
    )


def work(job):
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

    install_numpy_2_pickle_compat()
    import numpy.core.numeric, sys

    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    from drivesta_feedback.batch.actor_tail import extend_observation
    from drivesta_feedback.batch.batched_official import BatchedOfficial

    token, path, candidate_file, cutoff, end = job
    with lzma.open(path, "rb") as f:
        cache = pickle.load(f)
    nominal, alternate, meta = extend_observation(cache.observation, cutoff, end)
    row = dict(
        token=token,
        actor_count=meta["actor_count"],
        speed_envelope_actors=sum(a["alternative_is_speed_bound_envelope"] for a in meta["actors"]),
        nominal_geometry=geometry_error(cache.observation, nominal, cutoff, end),
        alternative_geometry=geometry_error(cache.observation, alternate, cutoff, end),
        model_counts={
            k: sum(a["selected"] == k for a in meta["actors"])
            for k in ("cv", "turn_accel", "static")
        },
    )
    if candidate_file:
        with np.load(candidate_file) as z:
            paths = z["trajectories"].copy()
            old = z["labels"].copy()
        original = BatchedOfficial(cache)
        full, _ = original.score(paths)
        if not np.allclose(old, full, atol=1e-10, rtol=0):
            raise ValueError("Full label reproduction failed")
        inferred = copy.copy(cache)
        inferred.observation = nominal
        inferred._alternative_observation = alternate
        scorer = BatchedOfficial(inferred)
        pred, _ = scorer.score(paths)
        alt = scorer.alternative_labels.copy()
        if not np.array_equal(pred[:, [0, 1, 2, 4, 5]], full[:, [0, 1, 2, 4, 5]]):
            raise ValueError("Actor tail changed a non-TTC component")
        good = pred[:, -1] >= 0.95
        robust = good & (alt[:, -1] >= 0.95)
        truth = full[:, -1] >= 0.95
        row.update(
            candidates=len(paths),
            ttc_mismatch=int(np.count_nonzero(pred[:, 3] != full[:, 3])),
            nominal_high=int(good.sum()),
            robust_high=int(robust.sum()),
            nominal_high_full_low=int(np.count_nonzero(good & ~truth)),
            robust_high_full_low=int(np.count_nonzero(robust & ~truth)),
            full_high=int(truth.sum()),
            full_high_robust_low=int(np.count_nonzero(truth & ~robust)),
            full_label_max_error=float(abs(old - full).max()),
            total_mae_points=float(100 * abs(pred[:, -1] - full[:, -1]).mean()),
            selected_model_disagreements=int(np.count_nonzero(pred[:, 3] != alt[:, 3])),
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    from drivesta_feedback.paths import bootstrap_navsim

    r = bootstrap_navsim()
    os.chdir(r)
    native = r / "feedback/runs/feedback_v2_design_20261019/native128_eval_aligned"
    from navsim.common.dataloader import MetricCacheLoader

    caches = MetricCacheLoader(r / "exp/navsim_cache/metric_cache")
    rows = json.loads((native / "summary.json").read_text())["rows"]
    real = [
        (
            x["token"],
            str(caches.metric_cache_paths[x["token"]]),
            str(native / (x["token"] + ".npz")),
            4.0,
            5.0,
        )
        for x in rows
    ]
    synth = [
        json.loads(s)
        for s in (r / "feedback/runs/simscale_physical_pilot_v1/scenes.jsonl")
        .read_text()
        .splitlines()
    ]
    synthetic = [(x["token"], x["source_cache_path"], None, 3.0, 4.0) for x in synth]
    args.output.mkdir(parents=True, exist_ok=True)
    result = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for name, jobs in [("real128", real), ("synthetic20", synthetic)]:
            rows = list(pool.map(work, jobs))
            result[name] = dict(scenes=len(rows), rows=rows)
            result[name]["speed_envelope_actors"] = sum(x["speed_envelope_actors"] for x in rows)
            for field in ("nominal_geometry", "alternative_geometry"):
                a = np.array([v for row in rows for v in row[field]["distances"]])
                b = np.array([v for row in rows for v in row[field]["final_distances"]])
                result[name][field] = dict(
                    actor_steps=len(a),
                    ade_m=float(a.mean()),
                    p95_m=float(np.quantile(a, 0.95)),
                    final_mean_m=float(b.mean()),
                    final_p95_m=float(np.quantile(b, 0.95)),
                    missing_actor_steps=sum(row[field]["missing_actor_steps"] for row in rows),
                )
            if name == "real128":
                for k in (
                    "candidates",
                    "ttc_mismatch",
                    "nominal_high",
                    "robust_high",
                    "nominal_high_full_low",
                    "robust_high_full_low",
                    "full_high",
                    "full_high_robust_low",
                    "selected_model_disagreements",
                ):
                    result[name][k] = sum(row[k] for row in rows)
                result[name]["total_mae_points"] = float(
                    np.mean([x["total_mae_points"] for x in rows])
                )
            print(
                name, json.dumps({k: v for k, v in result[name].items() if k != "rows"}), flush=True
            )
    result["scope"] = (
        "Diagnostic forecasts. Real128 is already-viewed NAVTEST; synthetic20 hides observed3..4s. No future truth used by forecaster. Geometry errors are centroid distances on matched actor-times, including static actors; unmatched births are counted separately. robust_high uses two PDM thresholds only, without production driving filters. Alternative geometry may be an uncertainty envelope rather than a point forecast."
    )
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
