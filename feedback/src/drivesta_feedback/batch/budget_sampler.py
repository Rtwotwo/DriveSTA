"""Deterministic, bounded exploration on a GT-anchored 0.30 m lattice."""

import hashlib
import numpy as np
from scipy.spatial import cKDTree


def grid(bounds, spacing, unit=0.05, offset=0.0):
    stride = int(round(spacing / unit))
    shift = int(round(offset / unit))
    axes = []
    for lo, hi in (bounds[:2], bounds[2:]):
        first = int(np.ceil((lo / unit - shift) / stride - 1e-9))
        last = int(np.floor((hi / unit - shift) / stride + 1e-9))
        axes.append(np.arange(first, last + 1, dtype=np.int32) * stride + shift)
    x, y = np.meshgrid(*axes, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel())).astype(np.int32)


def rng_for(token, stage):
    seed = int.from_bytes(hashlib.sha256((token + ":" + stage).encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def unmeasured(keys, measured):
    seen = {tuple(k) for k in measured}
    return np.asarray([k for k in keys if tuple(k) not in seen], dtype=np.int32).reshape(-1, 2)


def spread_order(keys, rng, tile_stride, priority=None):
    """Visit one point per coarse spatial tile before returning to a tile."""
    if not len(keys):
        return keys
    jitter = rng.random(len(keys))
    order = np.argsort(jitter) if priority is None else np.lexsort((jitter, -priority))
    tiles = np.floor_divide(keys, tile_stride)
    ranks = np.empty(len(keys), dtype=np.int32)
    counts = {}
    for i in order:
        tile = tuple(tiles[i])
        ranks[i] = counts.get(tile, 0)
        counts[tile] = int(ranks[i]) + 1
    order_rank = np.empty(len(keys), dtype=np.int32)
    order_rank[order] = np.arange(len(keys))
    return keys[np.lexsort((order_rank, ranks))]


def next_request(rows, completed_stages, config, token):
    """Return an immutable next phase; callers persist it before scoring."""
    c = config["sampling"]
    unit, spacing = c["integer_lattice_m"], c["dense_m"]
    left = c["max_unique_goals"] - len(rows)
    if left <= 0:
        return None
    measured = rows["keys"]
    tile = int(round(c["coarse_m"] / unit))
    phases = [
        ("base", grid(c["base_bounds_m"], spacing, unit)),
        ("coarse", grid(c["hard_bounds_m"], c["coarse_m"], unit)),
        ("offset", grid(c["hard_bounds_m"], c["coarse_m"], unit, c["offset_probe_m"])),
    ]
    for name, points in phases:
        if name in completed_stages:
            continue
        points = unmeasured(points, measured)
        points = spread_order(points, rng_for(token, name), tile)
        if name == "base" and len(points):
            # The zero-offset warp is a separately labelled candidate.
            zero = np.flatnonzero((points == 0).all(axis=1))
            if len(zero):
                points = np.concatenate((points[zero], np.delete(points, zero, axis=0)))
        limit = c["offset_probe_limit"] if name == "offset" else len(points)
        return dict(name=name, keys=points[: min(left, limit)].tolist())

    points = unmeasured(grid(c["hard_bounds_m"], spacing, unit), measured)
    if not len(points):
        return None
    name = "adaptive_%04d" % sum(s.startswith("adaptive_") for s in completed_stages)
    rng = rng_for(token, name)
    count = min(left, c["adaptive_batch"], len(points))
    distance, ids = cKDTree(measured * unit).query(points * unit, k=min(4, len(rows)))
    distance, ids = np.asarray(distance).reshape(len(points), -1), np.asarray(ids).reshape(
        len(points), -1
    )
    nearby_labels = rows["labels"][ids]
    best = nearby_labels[:, :, -1].max(axis=1)
    jump = np.ptp(nearby_labels, axis=1).max(axis=1)
    local = distance[:, 0] <= c["local_radius_m"]
    high = local & (best >= c["high_pdm_raw"])
    if not high.any() and (rows["labels"][:, -1] > 0).any():
        positive = rows["labels"][:, -1][rows["labels"][:, -1] > 0]
        high = local & (best >= np.quantile(positive, 0.9))
    boundary = local & (jump > c["component_jump_raw"])
    global_count = max(1, int(np.ceil(count * c["exploration_fraction"])))
    high_count = (count - global_count) // 2
    chosen = []
    for mask, quota, priority in (
        (boundary, count - global_count - high_count, jump),
        (high, high_count, best),
    ):
        indices = np.flatnonzero(mask)
        if chosen:
            occupied = {tuple(k) for k in chosen}
            indices = np.array([i for i in indices if tuple(points[i]) not in occupied], dtype=int)
        ordered = spread_order(points[indices], rng, tile, priority[indices])
        chosen.extend(ordered[:quota].tolist())
    remaining = unmeasured(points, chosen)
    chosen.extend(spread_order(remaining, rng, tile)[: count - len(chosen)].tolist())
    return dict(name=name, keys=chosen)


def coverage(rows, config):
    c = config["sampling"]
    domain = grid(c["hard_bounds_m"], c["dense_m"], c["integer_lattice_m"])
    remaining = unmeasured(domain, rows["keys"])
    high = rows["labels"][:, -1] >= c["high_pdm_raw"]
    limits = np.array([domain.min(axis=0), domain.max(axis=0)])
    edge = ((rows["keys"] == limits[0]) | (rows["keys"] == limits[1])).any(axis=1)
    open_high = 0
    if len(remaining) and high.any():
        dist, _ = cKDTree(remaining).query(rows["keys"][high], k=1)
        open_high = int((dist <= c["dense_m"] / c["integer_lattice_m"] + 1e-6).sum())
    return dict(
        domain_grid_points=len(domain),
        measured=len(rows),
        unmeasured=len(remaining),
        budget_reached=len(rows) >= c["max_unique_goals"],
        high_on_outer_boundary=int((high & edge).sum()),
        high_with_unmeasured_neighbors=open_high,
        complete_domain_grid=not len(remaining),
        dense_spacing_m=c["dense_m"],
    )
