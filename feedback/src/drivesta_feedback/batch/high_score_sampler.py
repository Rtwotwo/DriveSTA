"""Budgeted endpoint proposals; millimetre keys identify requests, not a dense grid."""

import hashlib
import numpy as np


def next_request(rows, config, token, budget):
    c = config["sampling"]
    n = len(rows)
    remaining = budget - n
    if remaining <= 0:
        return None
    count = min(c["adaptive_batch"], remaining)
    seed = int.from_bytes(hashlib.sha256((token + ":" + str(n)).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    unit = c["integer_lattice_m"]
    bounds = np.asarray(c["hard_bounds_m"]).reshape(2, 2)
    base = np.asarray(c["base_bounds_m"]).reshape(2, 2)
    seen = {tuple(k) for k in rows["keys"]}
    positive = np.flatnonzero(rows["labels"][:, -1] > 0)
    elite_ids = positive[np.argsort(-rows["labels"][positive, -1], kind="stable")[:32]]
    elites = rows["keys"][elite_ids] * unit
    keys = []
    for attempt in range(count * 1000):
        if len(keys) == count:
            break
        if n == 0:
            domain = base if len(keys) % 2 == 0 else bounds
            point = rng.uniform(domain[:, 0], domain[:, 1])
        elif len(keys) < int(np.ceil(count * c["exploration_fraction"])) or not len(elites):
            point = rng.uniform(bounds[:, 0], bounds[:, 1])
        else:
            scale = rng.choice(c["local_scales_m"])
            point = elites[rng.integers(len(elites))] + rng.normal(0, scale, 2)
        key = np.rint(point / unit).astype(np.int32)
        actual = key * unit
        if (actual < bounds[:, 0]).any() or (actual > bounds[:, 1]).any():
            continue
        if tuple(key) not in seen:
            seen.add(tuple(key))
            keys.append(key.tolist())
    if len(keys) != count:
        raise RuntimeError("Unable to produce requested number of distinct endpoints")
    return dict(name="search_" + str(n), keys=keys)
