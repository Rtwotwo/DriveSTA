"""Deterministic 1s actor-tail inference; measured occupancy is never edited."""

import copy
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class TailPolicy:
    version: str = "actor_tail_v1"
    history_s: float = 1.5
    holdout_s: float = 0.5
    recent_missing_s: float = 0.5
    stopped_speed_mps: float = 0.15
    max_vehicle_speed_mps: float = 40.0
    max_accel_mps2: float = 3.0
    max_decel_mps2: float = 6.0
    max_yaw_rate_rps: float = 0.6
    max_lateral_accel_mps2: float = 4.0
    accel_decay_s: float = 0.5
    turn_decay_s: float = 1.5


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def fit_motion(times, centers, angles, mode, policy, kind="VEHICLE", fallback_velocity=None):
    """Anchor forecast at the final observed pose, not a regression intercept."""
    times, centers = np.asarray(times, float), np.asarray(centers, float)
    angles = np.unwrap(np.asarray(angles, float))
    if not len(times) or not np.isfinite(centers).all() or not np.isfinite(angles).all():
        raise ValueError("Invalid actor history")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Actor timestamps must increase")
    static = kind not in ("VEHICLE", "PEDESTRIAN", "BICYCLE", "EGO", "UNKNOWN")
    t = times - times[-1]
    xy = centers - centers[-1]
    speed_cap = (
        4.0 if kind == "PEDESTRIAN" else 15.0 if kind == "BICYCLE" else policy.max_vehicle_speed_mps
    )
    v = np.asarray(fallback_velocity if fallback_velocity is not None else [0.0, 0.0], float)
    fallback_rejected = len(times) < 2 and (
        fallback_velocity is None or not np.isfinite(v).all() or np.linalg.norm(v) > speed_cap
    )
    if fallback_rejected:
        v = np.zeros(2)
    acc, omega = 0.0, 0.0
    if len(t) >= 2:
        recent = t >= -0.5 - 1e-8
        v = np.linalg.lstsq(t[recent, None], xy[recent], rcond=None)[0][0]
        last = t >= -0.4 - 1e-8
        if (
            len(t[last]) >= 3
            and np.max(np.linalg.norm(xy[last], axis=1)) < policy.stopped_speed_mps * 0.4
        ):
            static = True
        if mode == "turn_accel" and len(t) >= 6 and not static:
            v_a = np.linalg.lstsq(np.stack([t, 0.5 * t * t], -1), xy, rcond=None)[0]
            v = v_a[0]
            speed = np.linalg.norm(v)
            if speed > 0.3:
                acc = float(np.dot(v_a[1], v) / speed)
                angle_delta = angles - angles[-1]
                omega = float(np.linalg.lstsq(t[:, None], angle_delta, rcond=None)[0][0])
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite actor velocity")
    speed = min(float(np.linalg.norm(v)), speed_cap)
    angle = float(np.arctan2(v[1], v[0]))
    if static:
        speed, acc, omega = 0.0, 0.0, 0.0
    acc = float(np.clip(acc, -policy.max_decel_mps2, policy.max_accel_mps2))
    omega = float(
        np.clip(
            omega,
            -min(policy.max_yaw_rate_rps, policy.max_lateral_accel_mps2 / max(speed, 0.5)),
            min(policy.max_yaw_rate_rps, policy.max_lateral_accel_mps2 / max(speed, 0.5)),
        )
    )
    return dict(
        center=centers[-1].tolist(),
        speed=speed,
        direction=angle,
        accel=acc,
        yaw_rate=omega,
        max_speed=speed_cap,
        stationary=static,
        mode="static" if static else mode,
        fallback_rejected=bool(fallback_rejected),
    )


def predict_motion(model, future, policy):
    future = np.asarray(future, float)
    if (future < 0).any() or (len(future) > 1 and np.diff(future).min() < 0):
        raise ValueError("Prediction times must be nonnegative and ordered")
    # Small integration steps bound acceleration and stop without changing direction.
    p = np.asarray(model["center"], float).copy()
    speed = model["speed"]
    angle = model["direction"]
    now = 0.0
    rotation = 0.0
    positions = []
    rotations = []
    for end in future:
        while now < end - 1e-10:
            dt = min(0.05, float(end - now))
            mid = now + dt / 2
            a = model["accel"] * np.exp(-mid / policy.accel_decay_s)
            w = model["yaw_rate"] * np.exp(-mid / policy.turn_decay_s)
            w = float(
                np.clip(
                    w,
                    -policy.max_lateral_accel_mps2 / max(speed, 0.5),
                    policy.max_lateral_accel_mps2 / max(speed, 0.5),
                )
            )
            next_speed = float(np.clip(speed + a * dt, 0, model["max_speed"]))
            heading = angle + w * dt / 2
            p += (speed + next_speed) / 2 * dt * np.array([np.cos(heading), np.sin(heading)])
            speed = next_speed
            angle += w * dt
            rotation += w * dt
            now += dt
        positions.append(p.copy())
        rotations.append(rotation)
    return np.asarray(positions).reshape(-1, 2), np.asarray(rotations)


def choose_motion(times, centers, angles, policy, kind, fallback):
    models = [
        fit_motion(times, centers, angles, m, policy, kind, fallback) for m in ("cv", "turn_accel")
    ]
    errors = [None, None]
    selected = 0
    times = np.asarray(times, float)
    centers = np.asarray(centers, float)
    angles = np.asarray(angles, float)
    cut = times[-1] - policy.holdout_s
    train = times <= cut + 1e-8
    test = ~train
    if train.sum() >= 6 and test.sum() >= 3:
        for i, mode in enumerate(("cv", "turn_accel")):
            fitted = fit_motion(
                times[train], centers[train], angles[train], mode, policy, kind, fallback
            )
            pred, rot = predict_motion(fitted, times[test] - times[train][-1], policy)
            errors[i] = float(
                np.linalg.norm(pred - centers[test], axis=-1).mean()
                + 0.5 * np.abs(wrap(angles[train][-1] + rot - angles[test])).mean()
            )
        selected = int(errors[1] < errors[0] * 0.9 - 0.02)
    return (
        models[selected],
        models[1 - selected],
        dict(
            selected=models[selected]["mode"],
            history_points=len(times),
            backtest_errors=errors,
            uncertain=len(times) < 6 or (errors[0] is not None and min(errors) > 1.5),
        ),
    )


def extend_observation(observation, cutoff_s=4.0, end_s=5.0, policy=None):
    """Return nominal and alternative caches, with exact measured prefix.

    Actor predictions are independent of candidate ego trajectories. No lane
    branch, future actor birth, traffic-light change or interaction is invented.
    """
    from shapely.affinity import rotate, translate
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
        PDMOccupancyMap,
    )

    policy = policy or TailPolicy()
    dt = float(observation._sample_interval)
    cutoff = int(round(cutoff_s / dt))
    end = int(round(end_s / dt))
    if not np.isclose(cutoff * dt, cutoff_s) or not np.isclose(end * dt, end_s) or end <= cutoff:
        raise ValueError("Invalid tail horizon")
    prefix = [observation[i] for i in range(cutoff + 1)]
    start = max(0, cutoff - int(round(policy.history_s / dt)))
    history = {}
    for i in range(start, cutoff + 1):
        for token in prefix[i].tokens:
            poly = prefix[i][token]
            coords = np.asarray(poly.exterior.coords)
            center = np.array([poly.centroid.x, poly.centroid.y])
            edge = coords[1] - coords[0]
            angle = float(np.arctan2(edge[1], edge[0]))
            history.setdefault(token, []).append((i * dt, center, angle, poly))
    tokens = []
    nominal = []
    alternate = []
    metadata = []
    future = np.arange(cutoff + 1, end + 1) * dt
    for token, values in history.items():
        last = values[-1]
        if cutoff_s - last[0] > policy.recent_missing_s + 1e-8:
            continue
        obj = observation.unique_objects.get(token)
        if obj is None and observation.red_light_token not in token:
            raise ValueError("Actor geometry has no source identity: " + token)
        kind = getattr(getattr(obj, "tracked_object_type", None), "name", "STATIC")
        if observation.red_light_token in token:
            kind = "STATIC"
        velocity = getattr(obj, "velocity", None)
        fallback = [velocity.x, velocity.y] if velocity is not None else None
        times = [v[0] for v in values]
        xy = [v[1] for v in values]
        angles = [v[2] for v in values]
        model, other, detail = choose_motion(times, xy, angles, policy, kind, fallback)
        a, ar = predict_motion(model, future - last[0], policy)
        b, br = predict_motion(other, future - last[0], policy)

        def polygons(positions, rotations):
            return [
                translate(
                    rotate(last[3], float(w), origin=tuple(last[1]), use_radians=True),
                    xoff=float(p[0] - last[1][0]),
                    yoff=float(p[1] - last[1][1]),
                )
                for p, w in zip(positions, rotations)
            ]

        tokens.append(token)
        nominal.append(polygons(a, ar))
        other_polygons = polygons(b, br)
        # Some synthetic newborn tracks have invalid velocity metadata. Never
        # clip such a value into a fabricated40m/s forecast. Keep nominal at
        # last position and test a reachable occupancy envelope for acceptance.
        envelope = model["fallback_rejected"] and not model["stationary"]
        if envelope:
            other_polygons = [
                last[3].buffer(model["max_speed"] * float(t - last[0])) for t in future
            ]
        alternate.append(other_polygons)
        detail["alternative_is_speed_bound_envelope"] = envelope
        metadata.append(
            dict(
                token=token,
                kind=kind,
                last_observed_s=float(last[0]),
                model=model,
                alternative=other,
                **detail,
            )
        )

    def build(polygons):
        maps = list(prefix)
        for i in range(end - cutoff):
            maps.append(PDMOccupancyMap(tokens, [p[i] for p in polygons]))
        out = copy.copy(observation)
        out._occupancy_maps = maps
        out._global_to_local_idcs = list(range(end + 1))
        out._observation_samples = end
        out._observation_sample_res = 1
        return out

    return (
        build(nominal),
        build(alternate),
        dict(
            policy=asdict(policy),
            observed_end_s=cutoff_s,
            inferred_end_s=end_s,
            actor_count=len(tokens),
            actors=metadata,
            prefix_preserved=True,
            actor_motion_is_inferred=True,
        ),
    )
