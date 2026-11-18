"""Fixed GT-relative families: identity persists before any score-based choice."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import qmc


def family_parameters(seed=20261028):
    unit = qmc.Sobol(6, scramble=True, seed=seed).random_base2(12)[:3072] * 2 - 1
    # Endpoints lie on 0.05m regular grids. Interior offsets vanish at endpoints.
    grids = []
    for nx, ny in [(48, 48), (32, 24)]:
        x, y = np.meshgrid(
            np.arange(-nx // 2, nx // 2), np.arange(-ny // 2, ny // 2), indexing="ij"
        )
        grids.append(np.stack([x.ravel(), y.ravel()], -1) * 0.05)
    parameters = unit * np.array([0.0, 0.0, 0.05, 0.03, 0.02, 0.01])
    parameters[:, :2] = np.concatenate(grids)
    zero = int(np.flatnonzero((parameters[:2304, :2] == 0).all(1))[0])
    parameters[[0, zero]] = parameters[[zero, 0]]
    parameters[0] = 0.0  # Exact expert is an explicitly identified baseline.
    return parameters


def construct(expert, parameters):
    """C2 displacement about GT; preserve GT heading residual and exact zero."""
    expert = np.asarray(expert, dtype=np.float64)
    params = np.asarray(parameters, dtype=np.float64)
    if expert.shape != (8, 3) or params.ndim != 2 or params.shape[1] != 6:
        raise ValueError("Expected 8x3 expert and Nx6 family parameters")
    if not np.isfinite(expert).all() or not np.isfinite(params).all():
        raise ValueError("Nonfinite construction input")
    times = np.arange(9) * 0.5
    base = np.vstack([np.zeros(3), expert])
    u = times / 4
    beta = 10 * u**3 - 15 * u**4 + 6 * u**5
    middle = 64 * u**3 * (1 - u) ** 3
    basis = np.stack([beta, middle, middle * (2 * u - 1)], axis=-1)
    offsets = np.einsum("tk,nkd->ntd", basis, params.reshape(-1, 3, 2))
    tangent = np.stack([np.cos(base[:, 2]), np.sin(base[:, 2])], -1)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], -1)
    xy = base[None, :, :2] + offsets[..., :1] * tangent + offsets[..., 1:] * normal
    velocity = CubicSpline(times, xy, axis=1)(times, 1)
    base_velocity = CubicSpline(times, base[:, :2], axis=0)(times, 1)
    turn = np.arctan2(velocity[..., 1], velocity[..., 0]) - np.arctan2(
        base_velocity[:, 1], base_velocity[:, 0]
    )
    moving = (np.linalg.norm(velocity, axis=-1) >= 0.1) & (
        np.linalg.norm(base_velocity, axis=-1) >= 0.1
    )
    heading = base[None, :, 2] + np.where(moving, np.arctan2(np.sin(turn), np.cos(turn)), 0.0)
    result = np.concatenate(
        [xy[:, 1:], np.arctan2(np.sin(heading[:, 1:]), np.cos(heading[:, 1:]))[..., None]], -1
    )
    result[np.all(params == 0, axis=1)] = expert
    return result.astype(np.float32)


def local_reference(world_poses, initial_pose):
    result = np.asarray(world_poses, dtype=np.float64).copy()
    x, y, heading = initial_pose
    c, s = np.cos(heading), np.sin(heading)
    result[:, :2] = (result[:, :2] - [x, y]) @ np.array([[c, -s], [s, c]])
    result[:, 2] = np.arctan2(np.sin(result[:, 2] - heading), np.cos(result[:, 2] - heading))
    return result


def construct_bank(expert, pdm_reference, parameters):
    if np.asarray(parameters).shape != (3072, 6):
        raise ValueError("The fixed bank requires 3072 family parameters")
    return np.concatenate(
        [construct(expert, parameters[:2304]), construct(pdm_reference, parameters[2304:])]
    )


def unique_ranked(paths, scores, eligible, count=256):
    """Strict cutoff, descending score; no duplicated or near-identical padding."""
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError("Invalid EPDMS")
    ids = np.flatnonzero(np.asarray(eligible, bool) & (scores > 0.95))
    ids = ids[np.lexsort((ids, -scores[ids]))]
    keys, kept = set(), []
    for i in ids:
        key = np.round(np.asarray(paths[i]) * 1000).astype(np.int64).tobytes()
        if key in keys:
            continue
        keys.add(key)
        kept.append(int(i))
        if len(kept) == count:
            break
    return np.asarray(kept, dtype=np.int64)
