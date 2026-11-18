"""Classify every SimScale shortfall from saved queries and unchanged geometry."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import lzma
import multiprocessing as mp
from pathlib import Path
import pickle
import time
import traceback

import numpy as np


def load_cache(path):
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    from navsim.common.numpy_pickle_compat import install_numpy_2_pickle_compat

    install_numpy_2_pickle_compat()
    import numpy.core.numeric
    import sys

    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def classify(folder):
    from drivesta_feedback.batch.filter_targets import TargetPolicy, tracking_errors
    from drivesta_feedback.batch.physical_sampling import physical_mask, score_mask
    from drivesta_feedback.batch.adaptive_sampling import PreparedSpace
    from drivesta_feedback.batch.batched_official import BatchedOfficial
    import shapely

    folder = Path(folder)
    contract = json.loads((folder / "contract.json").read_text())
    constructor = contract["protocol"]["constructor"]
    policy = TargetPolicy(**constructor["policy"])
    result = json.loads((folder / "RESULT.json").read_text())
    rows = np.load(folder / "candidates.npy", allow_pickle=False)
    with np.load(folder / "construction.npz", allow_pickle=False) as z:
        labels, alternative = [rows["labels"]], [z["alternative_labels"].copy()]
        track, eligible = [z["tracking"].copy()], [z["eligible"].copy()]
    if (folder / "extra_queries.npz").exists():
        with np.load(folder / "extra_queries.npz", allow_pickle=False) as z:
            labels.append(z["labels"].copy())
            alternative.append(z["alternative_labels"].copy())
            track.append(z["tracking"].copy())
            eligible.append(z["eligible"].copy())
    y, alt, tracking, ok = map(np.concatenate, (labels, alternative, track, eligible))
    limits = [
        policy.max_tracking_ade_m,
        policy.max_tracking_error_m,
        policy.max_tracking_heading_error_rad,
    ]
    masks = [
        ("pdm95", y[:, -1] >= policy.min_pdm),
        ("nominal_components", score_mask(y, policy)),
        ("both_environments", score_mask(y, policy) & score_mask(alt, policy)),
        ("tracking", (tracking <= limits).all(1)),
        ("qualified", ok),
    ]
    remaining = np.ones(len(y), bool)
    funnel = {"queries": len(y)}
    for name, mask in masks:
        remaining &= mask
        funnel[name] = int(remaining.sum())
    cache = load_cache(contract["identity"]["official_cache_path"])
    space = PreparedSpace(cache)
    ego = cache.ego_state
    box = np.asarray(ego.car_footprint.geometry.exterior.coords)[:4]
    center = shapely.Point(ego.center.x, ego.center.y)
    initial_road = bool(shapely.covers(space.road, shapely.points(box)).all())
    initial_route = bool(shapely.covers(space.route, center))
    initial_valid = initial_road and initial_route
    witness = bool(space(np.zeros((8, 3), np.float32)))
    if initial_valid != witness or initial_valid != result["initial_space_valid"]:
        raise ValueError("Initial geometry disagrees with original production predicate")
    expert = np.asarray(constructor["expert"], np.float32)
    physical, physics_masks = physical_mask(
        expert[None],
        expert,
        policy,
        constructor["initial_velocity"],
        constructor["initial_acceleration"],
    )
    evaluator = BatchedOfficial(cache)
    gy, simulated = evaluator.score(expert[None])
    ga = evaluator.alternative_labels.copy()
    origin = [ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading]
    gt_track = np.asarray(tracking_errors(expert, simulated[0], origin))
    gt_space = bool(space(expert))
    gt_ok = bool(
        physical[0]
        and score_mask(gy, policy)[0]
        and score_mask(ga, policy)[0]
        and (gt_track <= limits).all()
        and gt_space
    )
    collision_time = evaluator.runtime.scorer.time_to_at_fault_collision(1)
    collision_time = float(collision_time) if np.isfinite(collision_time) else None
    if not initial_valid:
        category = (
            "initial_road_and_route"
            if not initial_road and not initial_route
            else "initial_road" if not initial_road else "initial_route"
        )
    else:
        category = "insufficient_qualified"
        for name in ("pdm95", "nominal_components", "both_environments", "tracking", "qualified"):
            if not funnel[name]:
                category = {
                    "qualified": "future_space",
                    "tracking": "tracking",
                    "both_environments": "actor_tail",
                    "nominal_components": "score_components",
                    "pdm95": "pdm_below95",
                }[name]
                break
    return dict(
        token=folder.name,
        source=result["source"],
        category=category,
        funnel=funnel,
        high_count=result["high_count"],
        best_pdms=float(y[:, -1].max() * 100),
        initial_road=initial_road,
        initial_route=initial_route,
        initial_road_max_outside_m=float(shapely.distance(shapely.points(box), space.road).max()),
        initial_route_distance_m=float(space.route.distance(center)),
        omitted_polygons=space.omitted,
        nominal_component_failures={
            name: int((y[:, i] < 1).sum())
            for i, name in enumerate(("NC", "DAC", "EP", "TTC", "comfort", "DDC"))
        },
        gt=dict(
            labels=gy[0].tolist(),
            alternative_labels=ga[0].tolist(),
            physics_failures=[k for k, v in physics_masks.items() if not v[0]],
            tracking=gt_track.tolist(),
            planned_space=gt_space,
            eligible=gt_ok,
            at_fault_collision_time_s=collision_time,
        ),
        source_result=result["config_digest"],
    )


def safe_classify(folder):
    try:
        return classify(folder)
    except Exception as e:
        from drivesta_feedback.batch.filter_targets import MissingPlannedGeometry

        if isinstance(e, MissingPlannedGeometry):
            folder = Path(folder)
            result = json.loads((folder / "RESULT.json").read_text())
            return dict(
                token=folder.name,
                source=result["source"],
                category="missing_geometry",
                high_count=result["high_count"],
                initial_road=False,
                initial_route=False,
                gt={},
                detail=str(e),
            )
        return dict(token=Path(folder).name, error=traceback.format_exc())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    tokens = a.tokens.read_text().split()
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate tokens")
    journal = a.output / "diagnostics.jsonl"
    rows = (
        {r["token"]: r for r in map(json.loads, journal.read_text().splitlines())}
        if journal.exists()
        else {}
    )
    pending = [t for t in tokens if t not in rows or "error" in rows[t]]
    start = time.time()
    with (
        ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool,
        journal.open("a") as f,
    ):
        futures = {pool.submit(safe_classify, str(a.root / "scenes/train" / t)): t for t in pending}
        for future in as_completed(futures):
            row = future.result()
            rows[row["token"]] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            if len(rows) % 128 == 0:
                print(
                    json.dumps(
                        dict(processed=len(rows), total=len(tokens), elapsed_s=time.time() - start)
                    ),
                    flush=True,
                )
    errors = [r for r in rows.values() if "error" in r]
    summary = dict(
        processed=len(rows),
        categories=dict(Counter(r["category"] for r in rows.values() if "error" not in r)),
        errors=errors,
        gt_qualified=sum(r.get("gt", {}).get("eligible", False) for r in rows.values()),
        elapsed_s=time.time() - start,
    )
    from drivesta_feedback.batch.store import write_json

    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("Audit incomplete; see token errors")


if __name__ == "__main__":
    main()
