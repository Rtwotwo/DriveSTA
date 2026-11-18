"""Offline multi-parent search with exact rollout feedback and fixed admission.

DriveTTO motivates reference-preserving residuals and achieved-state feedback.
Here the feedback is the official simulator, not the learned online scorer.
"""

from pathlib import Path
import hashlib
import numpy as np

from .store import write_npz, write_json
from .simscale_dataset import path_key
from .physical_sampling import physical_mask, score_mask
from .filter_targets import tracking_errors, wrap
from .adaptive_sampling import diverse_indices
from .recovery_seeds import to_local
from .recovery_rollout_search import Geometry, signed_clearance


def control_residual(parent, parameters, initial_velocity):
    """Transfer a bounded speed/yaw rollout difference to the stored reference.

    Zero residual returns the reference exactly, including at stationary poses.
    Clamping constructs proposals only; the unchanged physical screen follows.
    """
    parent = np.asarray(parent, np.float32)
    values = np.asarray(parameters, float).reshape(-1, 6)
    times = np.arange(41) * 0.1
    ref = np.vstack([np.zeros(3), parent])
    dense = np.stack(
        [
            np.interp(times, np.arange(9) * 0.5, np.unwrap(ref[:, 2]) if d == 2 else ref[:, d])
            for d in range(3)
        ],
        -1,
    )
    speeds = np.linalg.norm(np.diff(dense[:, :2], axis=0) / 0.1, axis=1)
    accel = np.diff(np.r_[np.linalg.norm(initial_velocity), speeds]) / 0.1
    yaw = np.diff(dense[:, 2]) / 0.1
    u = (times[1:] - 0.05) / 4
    basis = np.stack([np.ones(40), np.sin(np.pi * u), 2 * u - 1], -1)
    # Smooth onset avoids a large instantaneous acceleration change.
    basis *= np.minimum(times[1:] / 0.5, 1)[:, None]

    def rollout(c):
        v = np.full(len(c), np.linalg.norm(initial_velocity), float)
        h = np.zeros(len(c))
        xy = np.zeros((len(c), 2))
        rows = []
        da, dw = c[:, :3] @ basis.T, c[:, 3:] @ basis.T
        for k in range(40):
            new_v = np.clip(v + np.clip(accel[k] + da[:, k], -6.0, 6.0) * 0.1, 0.0, 25.0)
            w = np.clip(yaw[k] + dw[:, k], -1.0, 1.0)
            w = np.clip(
                w, -4 / np.maximum((v + new_v) / 2, 0.1), 4 / np.maximum((v + new_v) / 2, 0.1)
            )
            mid_h = h + w * 0.05
            xy += ((v + new_v) / 2 * 0.1)[:, None] * np.stack([np.cos(mid_h), np.sin(mid_h)], -1)
            h += w * 0.1
            v = new_v
            rows.append(np.column_stack([xy, h]).copy())
        return np.stack(rows, 1)[:, 4::5]

    nominal = rollout(np.zeros((1, 6)))
    residual = rollout(values) - nominal
    result = parent[None].astype(float) + residual
    result[:, :, 2] = wrap(result[:, :, 2])
    result = result.astype(np.float32)
    result[(values == 0).all(1)] = parent
    return result


def compensated(parent, tracked, gain):
    """Local inverse-tracking proposal. It never inherits the parent's labels."""
    result = np.asarray(parent, float).copy()
    delta = result - np.asarray(tracked, float)
    delta[:, 2] = wrap(delta[:, 2])
    result += gain * delta
    result[:, 2] = wrap(result[:, 2])
    return result.astype(np.float32)


def search_seeds(
    out,
    cfg,
    evaluator,
    space,
    expert,
    policy,
    velocity,
    acceleration,
    origin,
    old,
    parents,
    raw_expert,
):
    """Commit batches; restart from the same archive, RNG state and round."""
    from .simscale_recovery import gt_proposals

    settings = cfg["tto_seeds"]
    out = Path(out)
    checkpoint = out / "tto_seeds.npz"
    report = out / "tto_seeds.json"
    columns = ("trajectories", "labels", "alternative_labels", "tracking", "eligible")
    arrays = {k: np.empty((0,) + v.shape[1:], dtype=v.dtype) for k, v in old.items()}
    priorities = np.zeros(0)
    actual = np.zeros((0, 8, 3), np.float32)
    records = np.zeros((0, 8))
    rnd = 0
    proposed = 0
    pending = np.zeros((0, 8, 3), np.float32)
    pending_records = np.zeros((0, 8))
    next_batch = 0
    forbidden = {path_key(expert), path_key(raw_expert)}
    seed = int(hashlib.sha256(out.name.encode()).hexdigest()[:8], 16)
    limits = np.array(
        [
            policy.max_tracking_ade_m,
            policy.max_tracking_error_m,
            policy.max_tracking_heading_error_rad,
        ]
    )
    geometry = Geometry.__new__(Geometry)
    geometry.cache, geometry.road, geometry.route = space.cache, space.road, space.route
    geometry.vehicle, geometry.sampling = space.vehicle, space.sampling
    initial = np.zeros((0, 8, 3), np.float32)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as z:
            if "format_version" not in z or int(z["format_version"]) != 2:
                raise ValueError("Use a new output for legacy TTO seed checkpoints")
            if z["digest"].tobytes().decode() != cfg["digest"]:
                raise ValueError("TTO seed configuration changed")
            arrays = {k: z[k].copy() for k in columns}
            priorities, actual, records = (
                z["priority"].copy(),
                z["tracked"].copy(),
                z["records"].copy(),
            )
            rnd, proposed = int(z["round"]), int(z["proposed"])
            if "pending" in z:
                pending, pending_records, next_batch = (
                    z["pending"].copy(),
                    z["pending_records"].copy(),
                    int(z["next_batch"]),
                )
            if bool(z["finished"]):
                return arrays
    seen = {path_key(t) for t in arrays["trajectories"]}

    # Scoring a stored path in the seed archive is allowed, but final publication
    # deduplicates against the complete existing archive as well.
    def save(finished=False):
        write_npz(
            checkpoint,
            **arrays,
            priority=priorities,
            tracked=actual,
            records=records,
            format_version=np.array(2),
            initial_parents=initial,
            pending=pending,
            pending_records=pending_records,
            next_batch=np.array(next_batch),
            round=np.array(rnd),
            proposed=np.array(proposed),
            finished=np.array(finished),
            digest=np.frombuffer(cfg["digest"].encode(), np.uint8),
        )

    def score(paths, provenance):
        nonlocal arrays, priorities, actual, records, proposed
        paths = np.asarray(paths, np.float32)
        proposed += len(paths)
        physical, _ = physical_mask(paths, expert, policy, velocity, acceleration)
        near = (
            np.linalg.norm(paths[:, :, :2] - expert[None, :, :2], axis=-1).max(1)
            <= cfg["max_gt_deviation_m"]
        )
        ids = []
        for i in np.flatnonzero(physical & near):
            key = path_key(paths[i])
            if key not in seen and key not in forbidden:
                ids.append(i)
                seen.add(key)
        ids = ids[: settings["max_queries"] - len(arrays["labels"])]
        if not ids:
            return
        t = paths[ids]
        y, sim = evaluator.score(t)
        alt = evaluator.alternative_labels.copy()
        tracked = np.asarray([to_local(s[:, :3], origin)[5::5] for s in sim], np.float32)
        track = np.asarray([tracking_errors(p, s, origin) for p, s in zip(t, sim)])
        spatial = space.batch(t)
        good = score_mask(y, policy) & score_mask(alt, policy) & (track <= limits).all(1) & spatial
        # Continuous distance is an exploration signal. Exact official DDC and
        # unbuffered planned-space checks alone determine final acceptance.
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex

        centers = evaluator.runtime.scorer._ego_coords[1:, :, BBCoordsIndex.CENTER]
        clearance = signed_clearance(space.route, centers).reshape(len(t), -1)
        route_penalty = np.maximum(-clearance, 0).mean(1)
        component_penalty = (
            (y[:, [0, 1, 3, 4, 5]] < 1).sum(1) + (alt[:, [0, 1, 3, 4, 5]] < 1).sum(1)
        ) * 0.12
        priority = np.minimum(y[:, -1], alt[:, -1]) - component_penalty - 0.25 * route_penalty
        priority -= 0.2 * np.maximum(track / limits - 1, 0).sum(1) + 0.25 * (~spatial)
        priority += good.astype(float)
        rows = dict(trajectories=t, labels=y, alternative_labels=alt, tracking=track, eligible=good)
        arrays = {k: np.concatenate([v, rows[k]]) for k, v in arrays.items()}
        priorities = np.r_[priorities, priority]
        actual = np.concatenate([actual, tracked])
        records = np.concatenate([records, np.asarray(provenance)[ids]])

    # Existing diverse references are evaluated with continuous route feedback.
    old_priority = np.minimum(old["labels"][:, -1], old["alternative_labels"][:, -1])
    old_priority -= 0.2 * (old["labels"][:, [0, 1, 3, 4, 5]] < 1).sum(1)
    init_ids = diverse_indices(old["trajectories"], old_priority, 32, 0.1)
    initial = np.concatenate([parents[:32], old["trajectories"][init_ids]])
    while rnd < settings["rounds"] and len(arrays["labels"]) < settings["max_queries"]:
        rng = np.random.default_rng(seed + rnd)
        if len(pending):
            paths, metadata = pending, pending_records
        elif rnd == 0:
            paths = initial
            metadata = np.zeros((len(paths), 8))
            metadata[:, 0] = -1
            metadata[:, 1] = -np.arange(1, len(paths) + 1)
        else:
            ids = (
                diverse_indices(arrays["trajectories"], priorities, 12, 0.05)
                if len(priorities)
                else []
            )
            archive = arrays["trajectories"][ids] if ids else initial
            paths, metadata = [], []
            for k in range(settings["proposals_per_round"]):
                # A quarter of proposals continue to explore initial references.
                pidx = k % len(archive)
                parent = initial[k % len(initial)] if k % 4 == 0 else archive[pidx]
                parent_id = -(k % len(initial)) - 1 if k % 4 == 0 or not ids else ids[pidx]
                if not ids and k % 4 != 0:
                    parent_id = -pidx - 1
                qualified = bool(ids and arrays["eligible"][ids[pidx]]) and k % 4 != 0
                scale = (0.015, 0.05, 0.15)[k % 3] if qualified else (0.25, 0.6, 1.0)[(k // 4) % 3]
                op = (rnd + k // 4) % 3
                if op == 0:
                    c = rng.uniform(-1, 1, 6) * np.array([0.35, 2.0, 1.0, 1.0, 0.8, 0.15]) * scale
                    t = gt_proposals(parent, c[None])[0]
                elif op == 1:
                    c = rng.uniform(-1, 1, 6) * np.array([1.0, 1.0, 0.5, 0.12, 0.12, 0.08]) * scale
                    t = control_residual(parent, c[None], velocity)[0]
                elif ids and k % 4 != 0:
                    c = np.zeros(6)
                    c[0] = (0.25, 0.5, 0.75)[k % 3]
                    t = compensated(parent, actual[ids[pidx]], c[0])
                else:
                    c = rng.uniform(-1, 1, 6) * np.array([0.35, 2.0, 1.0, 1.0, 0.8, 0.15]) * scale
                    t = gt_proposals(parent, c[None])[0]
                    op = 0
                paths.append(t)
                metadata.append([op, parent_id, *c])
            paths = np.asarray(paths)
            metadata = np.asarray(metadata)
        # Store proposed rows and the cursor so a committed batch is never
        # reselected from a partially updated archive on resume.
        pending, pending_records = paths, metadata
        for start in range(next_batch, len(paths), cfg["score_batch"]):
            score(
                paths[start : start + cfg["score_batch"]],
                metadata[start : start + cfg["score_batch"]],
            )
            next_batch = start + cfg["score_batch"]
            save()
            if len(arrays["labels"]) >= settings["max_queries"]:
                break
        rnd += 1
        pending = np.zeros((0, 8, 3), np.float32)
        pending_records = np.zeros((0, 8))
        next_batch = 0
        save()
        if arrays["eligible"].sum() >= settings["seed_goal"]:
            break
    old_keys = {path_key(t) for t in old["trajectories"]}
    write_json(
        report,
        dict(
            queries=len(arrays["labels"]),
            duplicate_existing_queries=sum(path_key(t) in old_keys for t in arrays["trajectories"]),
            qualified=int(arrays["eligible"].sum()),
            rounds=rnd,
            proposed=proposed,
            thresholds_unchanged=True,
            feedback="official_tracking_and_scores",
            all_evaluated_rows_retained=True,
        ),
    )
    # A finished checkpoint must always have its accounting report available.
    save(True)
    return arrays
