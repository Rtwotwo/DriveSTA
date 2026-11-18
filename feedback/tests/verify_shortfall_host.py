"""Small official evaluator witness on an existing scored prefix; no sampling."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from drivesta_feedback.batch.source import load_scene, construct, Evaluator
from drivesta_feedback.batch.store import read_json, write_json
from drivesta_feedback.batch.retained_bank import read_full_scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    rows = []
    for token in ("85695eb2eb2655a3", "02af464315915a50", "022bd072ade05482"):
        directory = a.root / "scenes/train" / token
        metadata, bank = read_full_scene(directory)
        protocol = read_json(directory / "contract.json")["protocol"]
        record = {
            k: metadata[k]
            for k in ("token", "split", "log_name", "scene_type", "metric_cache_relative_path")
        }
        _, gt, cache, current = load_scene(record, [token], protocol)
        if current != metadata:
            raise ValueError("Host scene/cache metadata differs")
        paths, _ = construct(gt, bank["keys"][:64], protocol)
        if not np.array_equal(paths, bank["trajectories"][:64]):
            raise ValueError("Original 64-candidate constructor batch differs")
        ids = sorted(set([0, 1, 63, int(bank["labels"][:, -1].argmax())]))
        e = Evaluator(cache)
        start = time.monotonic()
        actual = np.stack([e.score(bank["trajectories"][i]) for i in ids])
        error = float(abs(actual - bank["labels"][ids]).max())
        if error > protocol["labeling"]["float64_parity_tolerance"]:
            raise ValueError("Official label parity failed")
        rows.append(
            dict(
                token=token,
                candidates=len(ids),
                max_official_error=error,
                constructor_batch64_equal=True,
                elapsed_s=time.monotonic() - start,
            )
        )
    result = dict(
        passed=True,
        host=platform.node(),
        python=sys.executable,
        torch=torch.__version__,
        numpy=np.__version__,
        checks=rows,
        updated_unix=time.time(),
    )
    write_json(a.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
