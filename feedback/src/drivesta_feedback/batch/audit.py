"""Whole-block validation and fresh official evaluator checks."""

from pathlib import Path
import numpy as np
from .store import load_blocks, read_json, write_json, write_npz, checksum


def audit_scene(directory, protocol, official=True):
    directory = Path(directory)
    summary = read_json(directory / "COMPLETE.json")
    contract = read_json(directory / "contract.json")
    metadata = contract["identity"]
    gt = np.asarray(metadata["expert_gt"])
    rows = []
    endpoint_error = 0.0
    composition_error = 0.0
    for phase in sorted((directory / "phases").iterdir()):
        completion = read_json(phase / "COMPLETE.json")
        for record in completion["blocks"]:
            if checksum(phase / record["file"]) != record["sha256"]:
                raise ValueError("Committed block checksum mismatch")
    for path, z in load_blocks(directory):
        goals = gt[-1, :2] + z["keys"] * protocol["sampling"]["integer_lattice_m"]
        endpoint_error = max(
            endpoint_error,
            float(np.linalg.norm(z["trajectories"][:, -1, :2] - goals, axis=1).max()),
        )
        u = np.linspace(0, 1, 8, dtype=np.float64)
        beta = 10 * u**3 - 15 * u**4 + 6 * u**5
        expected = gt[None, :, :2] + beta[None, :, None] * (goals - gt[-1, :2])[:, None, :]
        if np.max(np.abs(expected - z["trajectories"][:, :, :2])) > 8e-5:
            raise ValueError("Warp differs from independently computed quintic positions")
        labels = z["labels"]
        recomposed = (
            labels[:, 0]
            * labels[:, 1]
            * (5 * labels[:, 2] + 5 * labels[:, 3] + 2 * labels[:, 4])
            / 12
        )
        composition_error = max(composition_error, float(abs(recomposed - labels[:, -1]).max()))
        rows.append((path, z))
    keys = np.concatenate([z["keys"] for _, z in rows])
    labels = np.concatenate([z["labels"] for _, z in rows])
    paths = np.concatenate([z["trajectories"] for _, z in rows])
    if len(keys) != summary["candidates"] or len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Candidate coverage or uniqueness mismatch")
    if (
        endpoint_error >= protocol["constructor"]["max_planned_endpoint_error_m"]
        or composition_error > 1e-9
    ):
        raise ValueError("Endpoint/score formula mismatch")
    ids = set(
        [
            0,
            len(keys) - 1,
            int(np.argmax(labels[:, -1])),
            int(np.argmin(labels[:, -1])),
            int(np.argmin(abs(labels[:, -1] - 0.95))),
        ]
    )
    zero = np.flatnonzero((keys == 0).all(axis=1))
    if len(zero) != 1:
        raise ValueError("Expected exactly one zero-offset candidate")
    ids.add(int(zero[0]))
    ids.update(
        int(x)
        for x in np.random.default_rng(20261014).choice(
            len(keys), size=min(8, len(keys)), replace=False
        )
    )
    probes = []
    if official:
        from .source import load_scene, Evaluator

        _, aligned_gt, cache, _ = load_scene(metadata, [metadata["token"]], protocol)
        if not np.array_equal(gt, aligned_gt):
            raise ValueError("Expert trajectory changed")
        evaluator = Evaluator(cache)
        gt_labels, gt_rollout = evaluator.score(gt, details=True)
        gt_error = float(abs(gt_labels - read_json(directory / "expert.json")["labels"]).max())
        if gt_error > 1e-9:
            raise ValueError("GT score mismatch")
        rollouts = []
        for i in sorted(ids, reverse=True):
            values, rollout = evaluator.score(paths[i], details=True)
            error = float(abs(values - labels[i]).max())
            if error > protocol["labeling"]["float64_parity_tolerance"]:
                raise ValueError("Independent official detail entry point mismatch")
            probes.append(dict(index=i, integer_key=keys[i].tolist(), max_component_error=error))
            rollouts.append(rollout)
        write_npz(
            directory / "audit_rollouts.npz",
            indices=np.asarray(sorted(ids, reverse=True)),
            simulated_states=np.stack(rollouts),
            expert_simulated_states=gt_rollout,
        )
    result = dict(
        token=summary["token"],
        passed=True,
        candidates=len(keys),
        official_checks=len(probes),
        max_endpoint_error_m=endpoint_error,
        max_composition_error=composition_error,
        max_official_error=max([p["max_component_error"] for p in probes], default=0.0),
        probes=probes,
        official=official,
    )
    write_json(directory / "audit.json", result)
    return result
