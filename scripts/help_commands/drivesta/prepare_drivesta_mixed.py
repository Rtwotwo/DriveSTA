"""Build verified12-frame SimScale video index without rewriting dense data."""

import argparse, hashlib, multiprocessing as mp, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from navsim.planning.training.drivesta.data.refinement.mixed_dataset import synthetic_video_record, VIDEO_POLICY, digest


import os
from navsim.common.release_paths import metadata_json as json, metadata_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index",
        default=str(
            ROOT
            / "data/indexes/drivesta/mixed.json"
        ),
    )
    p.add_argument(
        "--output",
        default=str(
            ROOT / "outputs/indexes/drivesta_mixed/simscale_video8_index.json"
        ),
    )
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    spec = json.loads(Path(a.index).read_text())
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(
            "Index exists; inspect and reuse it, or choose a new output: " + str(output)
        )
    for k in ("navsim_index", "simscale_manifest", "excluded_tokens"):
        if digest(spec[k]) != spec[k + "_sha256"]:
            raise ValueError("Changed data source " + k)
    with open(spec["simscale_manifest"]) as f:
        records = [
            r for r in map(json.loads, f) if r["source_group"] in spec["source_groups"]
        ]
    if len(records) != 129409 or len({r["token"] for r in records}) != 129409:
        raise ValueError("Synthetic population mismatch")
    rows = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, (token, row) in enumerate(
            pool.map(synthetic_video_record, records, chunksize=32)
        ):
            rows[token] = row
            if (i + 1) % 4096 == 0:
                print("video_checked", i + 1, "/", len(records), flush=True)
    data = dict(
        frame_policy=VIDEO_POLICY,
        source_manifest_sha256=spec["simscale_manifest_sha256"],
        source_groups=spec["source_groups"],
        records=rows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(output)
    summary = dict(
        scenes=len(rows),
        valid=sum(r["video_valid"] for r in rows.values()),
        all12_images_present=sum(
            all(f["available"] for f in r["frames"]) for r in rows.values()
        ),
        windows=dict(Counter(len(r["video_clips"]) for r in rows.values())),
        index_sha256=digest(output),
    )
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
