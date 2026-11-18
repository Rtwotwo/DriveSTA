"""Bounded recovery against immutable, same-slot adjacent-frame contexts."""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import qmc
from .dataset import (
    read_scene,
    cache_for,
    FastArchiveScorer,
    admission,
    ec_pair,
    score_from_components,
    DTYPE,
    POLICY,
    atomic_npz,
    atomic_npy,
)
from .sampling import construct, local_reference, unique_ranked
from ..batch.physical_sampling import physical_mask
from ..batch import filter_targets as _filter_module, physical_sampling as _physical_module
from ..batch.store import write_json, checksum


def keys(paths):
    return [np.round(p * 1000).astype(np.int64).tobytes() for p in paths]


def perturb_many(seeds, coefficients):
    """Use the validated C2 constructor without averaging unrelated trajectories."""
    result = np.empty_like(seeds, dtype=np.float32)
    # Seeds repeat; grouping saves spline work and preserves exact zero.
    groups = {}
    for i, p in enumerate(seeds):
        groups.setdefault(p.tobytes(), []).append(i)
    for ids in groups.values():
        result[ids] = construct(seeds[ids[0]], coefficients[ids])
    return result


def timed_reference(expert, amount):
    t = np.arange(9) * 0.5
    u = t / 4
    base = np.vstack([np.zeros(3), expert]).astype(float)
    base[:, 2] = np.unwrap(base[:, 2])
    requested = t + amount * 64 * u**3 * (1 - u) ** 3
    if np.any(np.diff(requested) <= 0) or requested.min() < 0 or requested.max() > 4:
        raise ValueError("Invalid bounded time perturbation")
    path = CubicSpline(t, base)(requested)[1:]
    path[:, 2] = np.arctan2(np.sin(path[:, 2]), np.cos(path[:, 2]))
    return path.astype(np.float32)


def previous_reference(previous_states, gap, current_pose):
    """Recede a real previous rollout; extend only its generated ego tail.

    The unchanged previous rollout is still the EC witness. This function only
    proposes a new current plan, which must pass all official and physical tests.
    """
    t = np.arange(41) * 0.1
    wanted = np.arange(1, 9) * 0.5 + gap
    world = previous_states[:, :, :3].copy()
    world[:, :, 2] = np.unwrap(world[:, :, 2], axis=1)
    interp = CubicSpline(t, world, axis=1)
    query = interp(np.minimum(wanted, 4))
    beyond = np.maximum(wanted - 4, 0)
    # Constant terminal pose velocity, not an invented observed future.
    query += beyond[None, :, None] * interp(4, 1)[:, None, :]
    start = interp(gap)
    result = np.stack([local_reference(q, current_pose) for q in query])
    start_local = local_reference(start, current_pose)
    u = np.arange(1, 9) / 8
    beta = u**3 * (10 - 15 * u + 6 * u * u)
    result[:, :, :2] -= start_local[:, None, :2] * (1 - beta)[None, :, None]
    result[:, :, 2] = np.arctan2(np.sin(result[:, :, 2]), np.cos(result[:, :, 2]))
    return result.astype(np.float32)


def propose(round_id, bank, expert, pdm, engine, prev_states, gap, seed):
    n = len(bank)
    unit = qmc.Sobol(6, scramble=True, seed=seed + round_id).random_base2(12)[:n] * 2 - 1
    good = unique_ranked(bank["trajectories"], bank["epdms"], bank["admissible"], count=32)
    base = np.repeat(expert[None], n, axis=0)
    if round_id == 0 and len(good):
        base = bank["trajectories"][good[np.arange(n) % len(good)]]
        coef = unit * np.array([0.25, 0.18, 0.08, 0.06, 0.03, 0.02])
        name = "qualified_seed_local"
    elif round_id == 1 and prev_states is not None:
        p = engine.cache.ego_state.rear_axle
        base = previous_reference(prev_states, gap, [p.x, p.y, p.heading])
        coef = unit * np.array([0.5, 0.3, 0.12, 0.08, 0.04, 0.03])
        name = "previous_rollout_receding"
    elif round_id == 2:
        # Trackable realizations of GT/PDM; scores and physical checks are fresh.
        reference = np.stack([expert, pdm])
        p = engine.cache.ego_state.rear_axle
        realized = engine.simulate(reference)
        reference = np.stack(
            [local_reference(s[5::5, :3], [p.x, p.y, p.heading]) for s in realized]
        )
        base = reference[np.arange(n) % 2]
        coef = unit * np.array([1.0, 0.5, 0.2, 0.1, 0.06, 0.04])
        name = "tracked_reference_local"
    elif round_id == 3:
        reference = np.stack(
            [timed_reference(expert, a) for a in [-0.25, -0.12, 0, 0.12, 0.25]] + [pdm]
        )
        base = reference[np.arange(n) % len(reference)]
        coef = unit * np.array([1.5, 0.75, 0.3, 0.15, 0.08, 0.04])
        name = "bounded_gt_timing"
    elif round_id >= 4 and len(good):
        base = bank["trajectories"][good[np.arange(n) % len(good)]]
        coef = unit * np.array([0.06, 0.06, 0.03, 0.02, 0.01, 0.01])
        name = "qualified_seed_fine"
    else:
        base[np.arange(n) % 4 == 0] = pdm
        scale = (
            np.array([5.0, 2.5, 0.6, 0.3, 0.15, 0.08])
            if round_id >= 6
            else np.array([2.5, 1.2, 0.4, 0.2, 0.1, 0.05])
        )
        coef = unit * scale
        name = "gt_bounded_exploration"
    paths = perturb_many(base, coef)
    return paths, name


def search(root, row, work, rounds=6, baseline=None):
    """Never publish here. Each completed round is a recoverable query journal."""
    root, work = Path(root), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    from .recovery_store import original_folder

    source = (
        (Path(baseline) / row["split"] / row["token"])
        if baseline
        else original_folder(root, row["split"], row["token"])
    )
    info, original, _ = read_scene(source)
    source_digest = checksum(source / "candidates.npy")
    completed = None
    if (work / "result.json").exists():
        result = json.loads((work / "result.json").read_text())
        if result["source_digest"] != source_digest:
            raise ValueError("Recovery input changed")
        if result["recovered"] or result.get("requested_rounds", 6) >= rounds:
            completed = result
    bank = np.array(original)
    cache, cache_path = cache_for(row)
    engine = FastArchiveScorer(cache)
    expert = np.asarray(cache.human_trajectory.poses)
    pose = cache.ego_state.rear_axle
    pdm = local_reference(engine.ref[5::5, :3], [pose.x, pose.y, pose.heading])
    previous = None
    gap = None
    previous_identity = None
    if info["previous_token"]:
        prev_path = (
            (Path(baseline) / row["split"] / info["previous_token"])
            if baseline
            else original_folder(root, row["split"], info["previous_token"])
        )
        prev_info, prev_bank, _ = read_scene(prev_path)
        prev_cache, prev_cache_path = cache_for(prev_info)
        previous = FastArchiveScorer(prev_cache).simulate(prev_bank["trajectories"])
        gap = info["start_time"] - prev_info["start_time"]
        previous_identity = dict(
            token=prev_info["token"],
            candidate_sha256=checksum(prev_path / "candidates.npy"),
            cache_path=str(prev_cache_path),
            cache_sha256=checksum(prev_cache_path),
            gap_seconds=gap,
        )
    identity = dict(
        source_digest=source_digest,
        cache_path=str(cache_path),
        cache_sha256=checksum(cache_path),
        previous=previous_identity,
        metric_files={
            name: checksum(Path(__file__).parent / name) for name in ("fast.py", "scoring.py")
        },
        safety_files={
            Path(module.__file__).name: checksum(Path(module.__file__))
            for module in (_filter_module, _physical_module)
        },
    )
    identity_path = work / "context.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Scoring cache, context or metric code changed on resume")
    else:
        journals = list(work.glob("round_*.npz"))
        if journals and work.parent.parent.name != "pilot":
            raise ValueError("Legacy query journals need explicit pilot migration")
        if journals:
            write_json(
                work / "legacy_pilot_journal_binding.json",
                dict(
                    bound_unix=time.time(),
                    prior_schedule="six-round pilot; stored per-round operator retained",
                    files={p.name: checksum(p) for p in journals},
                ),
            )
        write_json(identity_path, identity)
    if completed is not None:
        return completed
    dyn = cache.ego_state.dynamic_car_state
    velocity = [dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y]
    accel = [dyn.rear_axle_acceleration_2d.x, dyn.rear_axle_acceleration_2d.y]
    started = time.time()
    hist = []
    queried = 0
    for r in range(rounds):
        existing = unique_ranked(bank["trajectories"], bank["epdms"], bank["admissible"])
        if len(existing) == 256:
            break
        journal = work / f"round_{r:02d}.npz"
        if journal.exists():
            with np.load(journal, allow_pickle=False) as z:
                slots = z["slots"]
                scored = z["scored"]
                round_info = json.loads(str(z["metadata"]))
        else:
            paths, name = propose(
                r, bank, expert, pdm, engine, previous, gap, int(row["token"][:8], 16)
            )
            physical, _ = physical_mask(paths, expert, POLICY, velocity, accel)
            # Do not spend simulator queries on already-invalid motion.
            slots = np.flatnonzero(physical)
            paths = paths[slots]
            scored = np.zeros(len(paths), dtype=DTYPE)
            if len(paths):
                raw, comp, states = engine.score(paths, chunk=128, return_states=True)
                ec = (
                    ec_pair(states, previous[slots], gap)
                    if previous is not None
                    else np.full(len(paths), np.nan)
                )
                score = (
                    score_from_components(comp, ec)
                    if previous is not None
                    else score_from_components(comp)
                )
                valid, phys, track, space, errors, funnel = admission(
                    cache, paths, raw, states, score, expert
                )
                scored["family_id"] = slots
                for key, value in [
                    ("trajectories", paths),
                    ("components_raw", raw),
                    ("components", comp),
                    ("ec", ec),
                    ("epdms", score),
                    ("admissible", valid),
                    ("physical", phys),
                    ("tracking", track),
                    ("space", space),
                ]:
                    scored[key] = value
            else:
                funnel = {}
            round_info = dict(
                round=r, operator=name, proposed=3072, physical_precheck=len(paths), funnel=funnel
            )
            atomic_npz(
                journal, slots=slots, scored=scored, metadata=np.array(json.dumps(round_info))
            )
        # Preserve every prior admissible slot; replace it only with a distinct,
        # equally or more highly scored admissible path.
        seen = set(keys(bank["trajectories"][bank["admissible"]]))
        for j, slot in enumerate(slots):
            candidate = scored[j]
            old = bank[slot]
            if not candidate["admissible"]:
                continue
            k = keys(candidate["trajectories"][None])[0]
            if k in seen:
                continue
            if old["admissible"] and candidate["epdms"] < old["epdms"]:
                continue
            if old["admissible"]:
                seen.discard(keys(old["trajectories"][None])[0])
            bank[slot] = candidate
            seen.add(k)
        count = len(unique_ranked(bank["trajectories"], bank["epdms"], bank["admissible"]))
        round_info = dict(round_info, selected_after=count)
        hist.append(round_info)
        queried += len(slots)
        write_json(
            work / "progress.json",
            dict(token=row["token"], rounds=hist, queries=queried, updated_unix=time.time()),
        )
    ids = unique_ranked(bank["trajectories"], bank["epdms"], bank["admissible"])
    atomic_npy(work / "best.npy", bank)
    result = dict(
        token=row["token"],
        split=row["split"],
        source_digest=source_digest,
        original_count=info["selected_count"],
        selected_count=len(ids),
        recovered=len(ids) == 256,
        queries=queried,
        rounds=hist,
        requested_rounds=rounds,
        elapsed_s=time.time() - started,
        previous_token=info["previous_token"],
        context="immutable original previous family slot",
        best_file_sha256=checksum(work / "best.npy"),
    )
    write_json(work / "result.json", result)
    return result
