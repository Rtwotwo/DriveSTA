"""Deterministic multi-resolution sampling in integer endpoint coordinates."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def grid(bounds, spacing, unit=0.05):
    lo, hi, bottom, top = np.rint(np.asarray(bounds) / unit).astype(int)
    stride = int(round(spacing / unit))
    x, y = np.arange(lo, hi + 1, stride), np.arange(bottom, top + 1, stride)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()]), x, y


def missing_keys(keys, rows, remaining):
    keys = np.unique(np.asarray(keys, dtype=np.int32).reshape(-1, 2), axis=0)
    fresh = np.asarray([k for k in keys if tuple(k) not in rows], dtype=np.int32).reshape(-1, 2)
    return fresh[: max(0, remaining)], len(fresh) > remaining


def field(rows, bounds, spacing, unit=0.05):
    keys, x, y = grid(bounds, spacing, unit)
    values = np.array([rows.get(tuple(k), np.full(7, np.nan)) for k in keys])
    return keys, x, y, values.reshape(len(y), len(x), 7)


def edges(rows, bounds, threshold, unit=0.05):
    b = np.rint(np.asarray(bounds) / unit).astype(int)
    high = [k for k, v in rows.items() if v[-1] >= threshold]
    return [
        sum(k[0] == b[0] for k in high),
        sum(k[0] == b[1] for k in high),
        sum(k[1] == b[2] for k in high),
        sum(k[1] == b[3] for k in high),
    ]


def expand_bounds(rows, bounds, config):
    counts = edges(rows, bounds, config["high_pdm_raw"], config["integer_lattice_m"])
    hard = config["hard_x_m"] + config["hard_y_m"]
    steps = [config["expansion_x_m"]] * 2 + [config["expansion_y_m"]] * 2
    result = list(bounds)
    for j, count in enumerate(counts):
        if count:
            result[j] = (
                max(hard[j], bounds[j] - steps[j])
                if j % 2 == 0
                else min(hard[j], bounds[j] + steps[j])
            )
    return result, counts


def distance(mask, spacing):
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(~mask) * spacing


def dense_keys(rows, bounds, config):
    unit, spacing = config["integer_lattice_m"], config["dense_m"]
    keys, x, y = grid(bounds, spacing, unit)
    shape = (len(y), len(x))
    high_seed, transition_seed = np.zeros(shape, bool), np.zeros(shape, bool)
    stride = round(spacing / unit)

    def seed(source, mask):
        for key in source:
            ix, iy = int(round((key[0] - x[0]) / stride)), int(round((key[1] - y[0]) / stride))
            if 0 <= ix < len(x) and 0 <= iy < len(y):
                mask[iy, ix] = True

    high = [k for k, v in rows.items() if v[-1] >= config["high_pdm_raw"]]
    fallback = False
    if not high:
        positive = sorted((k for k in rows if rows[k][-1] > 0), key=lambda k: (-rows[k][-1], k))
        high = positive[: int(np.ceil(len(positive) * 0.1))]
        fallback = bool(high)
    seed(high, high_seed)
    coarse, _, _, c = field(rows, bounds, config["coarse_m"], unit)
    transition = np.zeros(c.shape[:2], bool)
    for axis in (0, 1):
        a, b = np.take(c, range(c.shape[axis] - 1), axis=axis), np.take(
            c, range(1, c.shape[axis]), axis=axis
        )
        valid = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
        change = valid & (
            (np.max(np.abs(a - b), axis=-1) > config["component_jump_raw"])
            | ((a[..., -1] >= config["high_pdm_raw"]) != (b[..., -1] >= config["high_pdm_raw"]))
        )
        if axis == 0:
            transition[:-1] |= change
            transition[1:] |= change
        else:
            transition[:, :-1] |= change
            transition[:, 1:] |= change
    seed(coarse[transition.ravel()], transition_seed)
    mask = (distance(high_seed, spacing) <= config["high_buffer_m"] + 1e-8) | (
        distance(transition_seed, spacing) <= config["transition_buffer_m"] + 1e-8
    )
    return keys[mask.ravel()], dict(
        fallback_used=fallback,
        high_or_fallback_seeds=len(high),
        transition_seeds=int(transition.sum()),
    )


def boundary_keys(rows, bounds, config):
    _, x, y, values = field(rows, bounds, config["dense_m"], config["integer_lattice_m"])
    v = values[..., -1]
    corners = np.stack([v[:-1, :-1], v[1:, :-1], v[:-1, 1:], v[1:, 1:]])
    mask = (
        np.isfinite(corners).all(axis=0)
        & (corners.min(axis=0) < config["high_pdm_raw"])
        & (corners.max(axis=0) >= config["high_pdm_raw"])
    )
    iy, ix = np.nonzero(mask)
    stride = int(round(config["dense_m"] / config["integer_lattice_m"]))
    sub = int(round(config["boundary_m"] / config["integer_lattice_m"]))
    return np.array(
        [
            (x[i] + dx, y[j] + dy)
            for j, i in zip(iy, ix)
            for dx in range(0, stride + 1, sub)
            for dy in range(0, stride + 1, sub)
        ],
        dtype=np.int32,
    ).reshape(-1, 2)


def open_dense_keys(rows, bounds, config):
    """Revisit high cells adjacent to unmeasured dense lattice points."""
    stride = int(round(config["dense_m"] / config["integer_lattice_m"]))
    limits = np.rint(np.asarray(bounds) / config["integer_lattice_m"]).astype(int)
    result = []
    for (x, y), v in rows.items():
        if x % stride or y % stride or v[-1] < config["high_pdm_raw"]:
            continue
        for dx, dy in ((stride, 0), (-stride, 0), (0, stride), (0, -stride)):
            k = (x + dx, y + dy)
            if limits[0] <= k[0] <= limits[1] and limits[2] <= k[1] <= limits[3] and k not in rows:
                result.append(k)
    return np.asarray(result, dtype=np.int32).reshape(-1, 2)
