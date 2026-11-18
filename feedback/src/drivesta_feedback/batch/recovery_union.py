"""Combine scored recovery operators, preserving provenance and replaying labels."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--trials", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from .source import load_scene, Evaluator
    from .filter_targets import (
        TargetPolicy,
        preliminary_rules,
        planned_space_checker,
        tracking_errors,
    )
    from .store import read_json, write_json, atomic

    results = [read_json(d / "result.json") for d in args.trials]
    record = results[0]["record"]
    if any(r["record"] != record for r in results):
        raise ValueError("Cannot combine different scenes")
    policies = [read_json(d.parent.parent / "config.json")["policy"] for d in args.trials]
    if any(x != policies[0] for x in policies):
        raise ValueError("Trial policies differ")
    policy = TargetPolicy(**policies[0])
    folder = args.source / "scenes" / record["split"] / record["token"]
    contract = read_json(folder / "contract.json")
    _, gt, cache, identity = load_scene(record, [record["token"]], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Original scene changed")
    paths, labels, source_ids, query_ids, accepted, seen = [], [], [], [], [], {}
    for source_id, d in enumerate(args.trials):
        with np.load(d / "scored.npz", allow_pickle=False) as z:
            for i, path in enumerate(z["trajectories"]):
                key = np.where(path == 0, 0.0, path).astype("<f4").tobytes()
                label = z["labels"][i]
                if key in seen:
                    if not np.allclose(labels[seen[key]], label, atol=1e-9, rtol=0):
                        raise ValueError("Repeated trajectory has conflicting official labels")
                    continue
                seen[key] = len(paths)
                paths.append(path)
                labels.append(label)
                source_ids.append(source_id)
                query_ids.append(i)
                accepted.append(bool(z["generator_eligible"][i]))
    paths, labels, accepted = np.asarray(paths), np.asarray(labels), np.asarray(accepted)
    valid = np.flatnonzero(accepted)
    ids = valid[np.argsort(-labels[valid, -1], kind="stable")[:256]]
    audit_error = None
    if len(ids) == 256:
        evaluator = Evaluator(cache)
        checker = planned_space_checker(cache)
        pose = cache.ego_state.rear_axle
        origin = [pose.x, pose.y, pose.heading]
        dyn = cache.ego_state.dynamic_car_state
        v = [dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y]
        a = [dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y]
        audit_error = 0.0
        for i in ids:
            label, sim = evaluator.score(paths[i], details=True)
            audit_error = max(audit_error, float(abs(label - labels[i]).max()))
            masks = preliminary_rules(paths[i : i + 1], label[None], gt, policy, v, a)
            errors = tracking_errors(paths[i], sim, origin)
            if (
                not all(x[0] for x in masks.values())
                or not checker(paths[i])
                or any(
                    x > y
                    for x, y in zip(
                        errors,
                        [
                            policy.max_tracking_ade_m,
                            policy.max_tracking_error_m,
                            policy.max_tracking_heading_error_rad,
                        ],
                    )
                )
            ):
                raise ValueError("Union trajectory failed fresh driving checks")
        if audit_error > 1e-9:
            raise ValueError("Official label replay differs")
    output = args.output / "scenes" / record["token"]
    output.mkdir(parents=True, exist_ok=True)
    atomic(
        output / "scored.npz",
        lambda f: np.savez(
            f,
            trajectories=paths,
            labels=labels,
            generator_eligible=accepted,
            selected_indices=ids,
            source_trial_ids=source_ids,
            source_query_indices=query_ids,
            source_trials=np.asarray([str(d) for d in args.trials]),
            operator="union",
        ),
    )
    if len(ids) == 256:
        t, y = list(paths[ids]), list(labels[ids])
        unique = {p.tobytes() for p in t}
        old = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
        for row in old:
            if row["trajectories"].tobytes() not in unique:
                t.append(row["trajectories"].copy())
                y.append(row["labels"].copy())
                unique.add(row["trajectories"].tobytes())
            if len(t) == 3072:
                break
        if len(t) != 3072:
            raise ValueError("Not enough distinct full-bank paths")
        atomic(
            output / "training_arrays.npz",
            lambda f: np.savez(
                f,
                trajectories=t,
                labels=y,
                high_indices=np.arange(256),
                generator_eligible_high_indices=np.arange(256),
            ),
        )
    representatives = []
    for i in ids:
        if (
            not representatives
            or np.linalg.norm(paths[representatives, :, :2] - paths[i, :, :2], axis=-1)
            .mean(-1)
            .min()
            >= 0.25
        ):
            representatives.append(i)
    result = dict(
        record=record,
        token=record["token"],
        seed="verified_trial_union",
        operator="union",
        selected=len(ids),
        eligible=int(accepted.sum()),
        passed=len(ids) == 256,
        queries=len(paths),
        max_official_label_error=audit_error,
        minimum_selected_pdms=float(labels[ids, -1].min() * 100) if len(ids) else None,
        distinct_representatives_025m=len(representatives),
        published_to_live_bank=False,
        training_array_count=3072 if len(ids) == 256 else 0,
        trials=[str(d) for d in args.trials],
    )
    write_json(output / "result.json", result)
    write_json(
        args.output / "summary.json",
        dict(
            scenes=1,
            passed=int(result["passed"]),
            errors=[],
            results=[result],
            recovered_tokens=[record["token"]] if result["passed"] else [],
            published_to_live_bank=False,
        ),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
