"""Verify new direct storage using previously scored real trajectories only."""

import argparse
from pathlib import Path
import tempfile
import numpy as np
from drivesta_feedback.batch.store import read_json, write_json, load_blocks, checksum
from drivesta_feedback.batch.source import load_scene, Evaluator, construct
from drivesta_feedback.batch.scene_store import SceneTable, read_scene, validate_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = read_json(args.reference_run / "run.json")["protocol"]
    scene_dir = next(iter(sorted((args.reference_run / "scenes/train").iterdir())))
    identity = read_json(scene_dir / "contract.json")["identity"]
    _, gt, cache, _ = load_scene(identity, [identity["token"]], protocol)
    _, reference = next(load_blocks(scene_dir))
    ids = np.unique(
        [
            0,
            1,
            2,
            len(reference["keys"]) - 1,
            int(reference["labels"][:, -1].argmax()),
            int(reference["labels"][:, -1].argmin()),
        ]
    )
    keys = reference["keys"][ids]
    trajectories = reference["trajectories"][ids]
    expected = reference["labels"][ids]
    rebuilt, endpoint_error = construct(gt, keys, protocol)
    construction_error = float(abs(rebuilt - trajectories).max())
    if construction_error > 3e-5:
        raise ValueError("Constructor regression against previously scored trajectories")
    evaluator = Evaluator(cache)
    labels = np.stack([evaluator.score(t) for t in trajectories])
    errors = float(abs(labels - expected).max())
    audit = Evaluator(cache)
    detail_error = max(
        float(abs(audit.score(t, details=True)[0] - label).max())
        for t, label in zip(trajectories, labels)
    )
    if max(errors, detail_error) > 1e-9:
        raise ValueError("Official score regression")
    with tempfile.TemporaryDirectory() as d:
        table = SceneTable(d, dict(identity=identity), len(keys) + 8)
        table.begin(dict(name="official_regression", keys=keys.tolist()))
        table.append(keys, trajectories, labels)
        table.end()
        validate_rows(table.rows, gt, protocol)
        table.close()
        write_json(
            Path(d) / "COMPLETE.json",
            dict(
                passed=True, candidates=len(keys), data_digest=checksum(Path(d) / "candidates.npy")
            ),
        )
        _, rows = read_scene(d, verify=True)
        np.testing.assert_array_equal(rows["trajectories"], trajectories)
        np.testing.assert_array_equal(rows["labels"], expected)
        del rows
    result = dict(
        passed=True,
        reference_scene=identity["token"],
        previously_scored_trajectories=len(keys),
        max_official_error=errors,
        max_details_error=detail_error,
        max_constructor_difference=construction_error,
        max_endpoint_error_m=endpoint_error,
        new_scene_sampling_started=False,
        full_generation_started=False,
    )
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
