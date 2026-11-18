"""Score all offline PDM proposals against the unchanged source evaluator."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np


def scored_prefix(states):
    """Extract exactly t=0..4s from an official 0.1s proposal buffer."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] < 41 or states.shape[1] < 3:
        raise ValueError("Proposal does not cover four seconds")
    result = states[:41, :3]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite proposal prefix")
    return result


def recover(job):
    record, config = job
    from .source import load_scene, Evaluator
    from .filter_targets import (
        TargetPolicy,
        preliminary_rules,
        planned_space_checker,
        tracking_errors,
    )
    from .recovery_seeds import to_local
    from .recovery_motion_search import joined_reference
    from .store import read_json, write_json, atomic
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
    from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
        BatchIDMPolicy,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    import os, torch

    torch.set_num_threads(1)
    token = record["token"]
    out = Path(config["output"]) / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    if (out / (token + ".json")).exists():
        return read_json(out / (token + ".json"))
    contract = read_json(
        Path(config["source"]) / "scenes" / record["split"] / token / "contract.json"
    )
    scene, expert, cache, identity = load_scene(record, [token], contract["protocol"])
    if identity != contract["identity"]:
        raise ValueError("Source identity changed")
    scenario = NavSimScenario(
        scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0"
    )
    inputs, initialization = MetricCacheProcessor(None, False)._get_planner_inputs(scenario)
    evaluator = Evaluator(cache)
    space = planned_space_checker(cache)
    policy = TargetPolicy(**config["policy"])
    pose = cache.ego_state.rear_axle
    origin = [pose.x, pose.y, pose.heading]
    dyn = cache.ego_state.dynamic_car_state
    v = np.array([dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y])
    a = np.array([dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y])
    paths = []
    labels = []
    names = []
    arms = {}
    seen = set()
    metadata = []
    for acceleration in config["accelerations"]:
        planner = PDMClosedPlanner(
            TrajectorySampling(num_poses=50, interval_length=0.1),
            TrajectorySampling(num_poses=40, interval_length=0.1),
            BatchIDMPolicy(
                speed_limit_fraction=config["speed_fractions"],
                fallback_target_velocity=15.0,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=float(acceleration),
                decel_max=3.0,
            ),
            lateral_offsets=config["lateral_offsets"],
            map_radius=100,
        )
        capture = []
        original = planner._generator.generate_proposals

        def generate(*args, **kwargs):
            values = original(*args, **kwargs)
            capture.append(values.copy())
            return values

        planner._generator.generate_proposals = generate
        planner.initialize(initialization)
        planner.compute_planner_trajectory(inputs)
        raw = capture[0]
        tracked = planner._scorer._states.copy()
        metadata.append(
            dict(
                acceleration=acceleration,
                proposals=len(raw),
                raw_array_shape=list(raw.shape),
                scored_horizon_s=4.0,
                route_matches_original=set(planner._route_lane_dict) == set(cache.route_lane_ids),
            )
        )
        if not metadata[-1]["route_matches_original"]:
            raise ValueError("Proposal source route differs")
        for i in range(len(raw)):
            # The generator allocates the five-second output horizon. Only its
            # first 41 states belong to the four-second scoring proposals.
            local = to_local(scored_prefix(raw[i]), origin)
            variants = [
                ("raw", local[5::5]),
                ("tracked", to_local(scored_prefix(tracked[i])[5::5], origin)),
                (
                    "joined",
                    joined_reference(np.arange(41) * 0.1, local, v, a, 3.0)[5::5].astype(
                        np.float32
                    ),
                ),
            ]
            for kind, path in variants:
                path = np.asarray(path, np.float32)
                key = path.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                label, sim = evaluator.score(path, details=True)
                errors = tracking_errors(path, sim, origin)
                masks = preliminary_rules(path[None], label[None], expert, policy, v, a)
                failed = [k for k, m in masks.items() if not m[0]]
                spatial = space(path)
                if not spatial:
                    failed.append("planned_space")
                if np.any(
                    np.asarray(errors)
                    > [
                        policy.max_tracking_ade_m,
                        policy.max_tracking_error_m,
                        policy.max_tracking_heading_error_rad,
                    ]
                ):
                    failed.append("tracking")
                name = "a%.1f_%03d_%s" % (acceleration, i, kind)
                arms[name] = dict(
                    pdms=float(label[-1] * 100),
                    labels=label.tolist(),
                    failed_rules=failed,
                    generator_eligible=not failed,
                    tracking_errors=list(errors),
                    planned_space_ok=spatial,
                )
                names.append(name)
                paths.append(path)
                labels.append(label)
    atomic(
        out / (token + ".npz"),
        lambda f: np.savez(
            f,
            names=np.asarray(names, dtype="U80"),
            trajectories=np.asarray(paths, np.float32),
            labels=np.asarray(labels, float),
        ),
    )
    row = dict(
        token=token,
        record=record,
        arms=arms,
        queries=len(paths),
        eligible=sum(not a["failed_rules"] for a in arms.values()),
        proposal_sources=metadata,
        no_candidate_score_inherited=True,
        source_modified=False,
    )
    write_json(out / (token + ".json"), row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokens", "policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    import yaml
    from .store import read_json, write_json

    tokens = a.tokens.read_text().split()
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid token list")
    records = {
        r["token"]: r for r in map(json.loads, (a.source / "scenes.jsonl").read_text().splitlines())
    }
    config = dict(
        version=2,
        source=str(a.source),
        output=str(a.output),
        tokens=sorted(tokens),
        policy=yaml.safe_load(a.policy.read_text()),
        accelerations=[1.5, 2.3],
        speed_fractions=[0.2, 0.4, 0.6, 0.8, 1.0],
        lateral_offsets=[-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
    )
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "config.json").exists() and read_json(a.output / "config.json") != config:
        raise ValueError("Resume differs")
    write_json(a.output / "config.json", config)
    results = []
    errors = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        jobs = {pool.submit(recover, (records[t], config)): t for t in tokens}
        for f in as_completed(jobs):
            try:
                row = f.result()
                results.append(row)
                print(json.dumps({k: row[k] for k in ("token", "queries", "eligible")}), flush=True)
            except Exception:
                error = dict(token=jobs[f], traceback=traceback.format_exc())
                errors.append(error)
                print(json.dumps(error), flush=True)
    summary = dict(
        scenes=len(results),
        queries=sum(r["queries"] for r in results),
        errors=errors,
        eligible_tokens=[r["token"] for r in results if r["eligible"]],
        elapsed_s=time.time() - start,
    )
    write_json(a.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("Proposal recovery errors")


if __name__ == "__main__":
    main()
