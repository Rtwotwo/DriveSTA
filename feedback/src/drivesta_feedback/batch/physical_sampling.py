"""Deterministic, bounded trajectory proposals and physical prechecks."""

from dataclasses import asdict
import numpy as np


def proposal_parameters(count, seed):
    from scipy.stats import qmc

    sampler = qmc.Sobol(6, scramble=True, seed=int(seed))
    return sampler.random_base2(int(np.ceil(np.log2(max(count, 1)))))[:count] * 2 - 1


def perturb(reference, coefficients):
    """C2 endpoint and interior offsets in the reference's local heading frame.

    Coefficients are metres, quantized to millimetres before construction.
    Changes and their first two derivatives vanish at t=0. The stored 0.5s
    plan is checked again; smooth analytic construction alone is insufficient.
    """
    from scipy.interpolate import CubicSpline

    ref = np.asarray(reference, float)
    coeff = np.asarray(coefficients, float).reshape(-1, 6)
    t = np.arange(9) * 0.5
    base = np.vstack([np.zeros(3), ref])
    u = t / 4
    endpoint = 10 * u**3 - 15 * u**4 + 6 * u**5
    middle = 64 * u**3 * (1 - u) ** 3
    earlylate = middle * (2 * u - 1)
    basis = np.stack([endpoint, middle, earlylate], -1)
    tangent = np.array([np.cos(ref[-1, 2]), np.sin(ref[-1, 2])])
    normal = np.array([-tangent[1], tangent[0]])
    offset = np.einsum("tk,nkd->ntd", basis, coeff.reshape(-1, 3, 2))
    xy = base[None, :, :2] + offset[..., :1] * tangent + offset[..., 1:] * normal
    # At a stationary endpoint retain the stored orientation instead of atan2(0,0).
    velocity = CubicSpline(t, xy, axis=1)(t, 1)
    heading = np.arctan2(velocity[..., 1], velocity[..., 0])
    heading = np.where(np.linalg.norm(velocity, axis=-1) < 0.05, base[None, :, 2], heading)
    return np.concatenate([xy[:, 1:], heading[:, 1:, None]], axis=-1).astype(np.float32)


def physical_mask(paths, expert, policy, velocity, acceleration):
    from .filter_targets import preliminary_rules

    # Ones only bypass label-dependent predicates. No synthetic label is saved.
    masks = preliminary_rules(
        paths, np.ones((len(paths), 7)), expert, policy, velocity, acceleration
    )
    masks = {
        k: v
        for k, v in masks.items()
        if k not in ("pdm", "safety_components", "direction_compliance")
    }
    return np.logical_and.reduce(list(masks.values())), masks


def score_mask(labels, policy):
    labels = np.asarray(labels)
    return (labels[:, -1] >= policy.min_pdm) & (labels[:, [0, 1, 3, 4, 5]] == 1).all(1)


def safe_planned_space_checker(cache):
    """Exclude malformed map polygons from the ADDITIONAL spatial screen.

    The official scoring map stays intact. Excluding polygons can only shrink
    the admitted region; no buffer, coordinate shift or route expansion occurs.
    """
    import copy
    from .filter_targets import planned_space_checker
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
        PDMDrivableMap,
    )

    m = cache.drivable_area_map
    valid = [i for i, t in enumerate(m.tokens) if m[t].is_valid and not m[t].is_empty]
    omitted = [t for i, t in enumerate(m.tokens) if i not in valid]
    if not omitted:
        return planned_space_checker(cache), omitted
    screened = copy.copy(cache)
    screened.drivable_area_map = PDMDrivableMap(
        [m.tokens[i] for i in valid],
        [m.map_types[i] for i in valid],
        [m[m.tokens[i]] for i in valid],
    )
    return planned_space_checker(screened), omitted


def select_bank(paths, labels, eligible, full_count=3072, high_count=256):
    if len(paths) != len(labels) or len(paths) != len(eligible):
        raise ValueError("Selection arrays do not align")
    high = np.flatnonzero(eligible)
    high = high[np.argsort(-labels[high, -1], kind="stable")][:high_count]
    rest = np.flatnonzero(~np.isin(np.arange(len(paths)), high))
    full = np.r_[high, rest[: max(0, full_count - len(high))]]
    return full, np.arange(len(high), dtype=np.int64)
