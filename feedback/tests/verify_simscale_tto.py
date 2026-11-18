"""Real-cache interrupted-batch equivalence; outputs stay inside the main bank."""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    from drivesta_feedback.paths import bootstrap_navsim

    root = bootstrap_navsim()
    from drivesta_feedback.batch import simscale_tto_seeds as search
    from drivesta_feedback.batch.store import read_json, write_json
    from drivesta_feedback.batch.filter_targets import TargetPolicy
    from drivesta_feedback.batch.batched_official import BatchedOfficial
    from drivesta_feedback.batch.adaptive_sampling import PreparedSpace
    from drivesta_feedback.audit.simscale_recovery import load_cache
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    token = "1e0747dde1595590-003"
    work = a.work / "resume_check_v2"
    work.mkdir(parents=True, exist_ok=True)
    c = read_json(root / "exp/drivesta_feedback_simscale/scenes/train" / token / "contract.json")
    constructor = c["protocol"]["constructor"]
    record = read_json(
        root
        / "feedback/runs/simscale_route_direction_20261021/directed_caches/scenes"
        / token
        / "repair.json"
    )
    cache = load_cache(record["cache_path"])
    cfg = yaml.safe_load((root / "feedback/configs/simscale_tto_recovery.yaml").read_text())
    cfg["tto_seeds"].update(max_queries=256, rounds=4, seed_goal=256)
    cfg["digest"] = "real_interrupted_batch_equivalence_v1"
    with np.load(a.work / "pilot32/queries" / token / "relabeled.npz") as z:
        old = {
            k: z[k].copy()
            for k in ("trajectories", "labels", "alternative_labels", "tracking", "eligible")
        }
    with np.load(a.work / "pilot32/queries" / token / "checkpoint.npz") as z:
        parents = z["parents"].copy()

    def run(folder):
        folder.mkdir(parents=True, exist_ok=True)
        return search.search_seeds(
            folder,
            cfg,
            BatchedOfficial(cache),
            PreparedSpace(cache),
            np.asarray(constructor["expert"], np.float32),
            TargetPolicy(**constructor["policy"]),
            constructor["initial_velocity"],
            constructor["initial_acceleration"],
            c["identity"]["ego_pose"],
            old,
            parents,
            np.asarray(c["identity"]["raw_expert_gt"], np.float32),
        )

    continuous = work / "continuous" / token
    resumed = work / "interrupted" / token
    run(continuous)
    original = search.write_npz
    calls = 0

    def interrupt(path, **kwargs):
        nonlocal calls
        original(path, **kwargs)
        calls += 1
        if calls == 3:
            raise RuntimeError("TEST_INTERRUPT_AFTER_COMMIT")

    search.write_npz = interrupt
    try:
        run(resumed)
    except RuntimeError as error:
        if str(error) != "TEST_INTERRUPT_AFTER_COMMIT":
            raise
    finally:
        search.write_npz = original
    if calls != 3:
        raise AssertionError("Interruption was not exercised")
    run(resumed)
    with np.load(continuous / "tto_seeds.npz") as x, np.load(resumed / "tto_seeds.npz") as y:
        checks = {k: bool(np.array_equal(x[k], y[k])) for k in x.files}
        assert x.files == y.files and all(checks.values()), checks
        write_json(
            work / "verification.json",
            dict(
                passed=True,
                token=token,
                scored=len(x["labels"]),
                interruption_after_commits=calls,
                arrays_equal=checks,
                official_real_cache=True,
            ),
        )
    print(json.dumps(read_json(work / "verification.json")))


if __name__ == "__main__":
    main()
