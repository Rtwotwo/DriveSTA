"""Self-contained construction records for published mixed-reference banks."""

import hashlib
import numpy as np


def record_digest(path, labels):
    return hashlib.sha256(
        np.asarray(path, dtype="<f4").tobytes() + np.asarray(labels, dtype="<f8").tobytes()
    ).hexdigest()


def reconstruct(group, indices, exact=False):
    seed = np.asarray(group["seed"], dtype=np.float32)
    keys = np.asarray(group["offset_keys"], dtype=np.int32)[indices]
    operator = group["operator"]
    if seed.shape != (8, 3) or not np.isfinite(seed).all():
        raise ValueError("Invalid construction reference")
    if operator == "expert_quintic":
        u = np.linspace(0, 1, 8)
        beta = 10 * u**3 - 15 * u**4 + 6 * u**5
        # The original expert reference was float64 before endpoint quantization.
        seed = np.asarray(group["seed"], dtype=np.float64)
        return seed[None, :, :2] + beta[None, :, None] * keys[:, None, :] * group["unit_m"]
    if operator == "endpoint":
        if keys.shape[1] != 2:
            raise ValueError("Invalid endpoint coefficients")
        if exact:
            from .source import construct
            from .gt_neighborhood import relative_heading

            protocol = dict(
                sampling={"integer_lattice_m": 0.001},
                constructor={"max_planned_endpoint_error_m": 3e-5},
            )
            zero = construct(seed, np.zeros((64, 2), np.int32), protocol)[0][0]
            return relative_heading(construct(seed, keys, protocol)[0], zero, seed, keys)
        u = np.linspace(0, 1, 8)
        beta = 10 * u**3 - 15 * u**4 + 6 * u**5
        return seed[None, :, :2] + beta[None, :, None] * keys[:, None, :] * 0.001
    if operator in ("interior", "paired_interior", "paired_interior_fine"):
        from .recovery_seed_pilot import interior_paths, paired_interior_paths

        width = 2 if operator == "interior" else 4
        if keys.shape[1] != width:
            raise ValueError("Invalid interior coefficients")
        return (interior_paths if width == 2 else paired_interior_paths)(seed, keys)
    raise ValueError("Unknown recovered constructor: " + operator)


def validate_mixed_rows(rows, constructor):
    if constructor.get("key_semantics") != "construction_group_and_query_index":
        raise ValueError("Missing mixed-bank key semantics")
    groups = constructor["groups"]
    keys = rows["keys"]
    if (keys < 0).any() or (keys[:, 0] >= len(groups)).any():
        raise ValueError("Unknown source construction")
    for group_id in np.unique(keys[:, 0]):
        mask = keys[:, 0] == group_id
        indices = keys[mask, 1]
        group = groups[int(group_id)]
        if (indices >= len(group["offset_keys"])).any():
            raise ValueError("Unknown source query")
        expected = reconstruct(group, indices)
        actual = rows["trajectories"][mask]
        if expected.shape[-1] == 2:
            if np.max(abs(actual[..., :2] - expected)) > 8e-5:
                raise ValueError("Recovered XY differs from source construction")
        elif not np.array_equal(actual, expected):
            raise ValueError("Recovered path differs from source construction")
        for i, row in zip(indices, rows[mask]):
            if (
                record_digest(row["trajectories"], row["labels"])
                != group["record_digests"][str(int(i))]
            ):
                raise ValueError("Published path or labels differ from verified source")
    y = rows["labels"]
    total = y[:, 0] * y[:, 1] * (5 * y[:, 2] + 5 * y[:, 3] + 2 * y[:, 4]) / 12
    if np.max(abs(total - y[:, 6])) > 1e-9:
        raise ValueError("Stored total differs from official PDM components")
