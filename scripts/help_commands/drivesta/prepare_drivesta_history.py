"""Prepare checked causal image paths; never write trajectory or score data."""

import argparse

import pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from navsim.common.release_paths import metadata_json as json, metadata_sha256


def read_log(job):
    log, group, logs, images = job
    with (Path(logs) / (log + ".pkl")).open("rb") as f:
        frames = pickle.load(f)
    lookup = {frame["token"]: i for i, frame in enumerate(frames)}
    manifest = {}
    errors = []
    for row in group:
        token = row["token"]
        i = lookup.get(token, -1)
        if i < 3:
            errors.append(dict(token=token, reason="missing_history"))
            continue
        offsets = [
            (f["timestamp"] - frames[i]["timestamp"]) / 1e6
            for f in frames[i - 1 : i + 1]
        ]
        if not offsets[0] < 0 or offsets[1] != 0:
            errors.append(
                dict(token=token, reason="noncausal_timestamp", offsets=offsets)
            )
            continue
        paths = [
            (Path(images) / f["cams"]["CAM_F0"]["data_path"]).resolve()
            for f in frames[i - 1 : i + 1]
        ]
        if not all(x.is_file() for x in paths):
            errors.append(dict(token=token, reason="missing_image"))
            continue
        manifest[token] = dict(
            image_paths=list(map(str, paths)), offsets=offsets, log_name=log
        )
    return manifest, errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index",
        default=str(
            ROOT / "data/indexes/drivesta/proposals.json"
        ),
    )
    p.add_argument(
        "--logs",
        default=str(
            Path(os.environ.get("OPENSCENE_DATA_ROOT", ROOT / "data"))
            / "navsim_logs/trainval"
        ),
    )
    p.add_argument(
        "--images",
        default=str(
            Path(os.environ.get("OPENSCENE_DATA_ROOT", ROOT / "data"))
            / "sensor_blobs/trainval"
        ),
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "outputs/indexes/drivesta_history/history_index.json"),
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--describe", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    source = json.loads(Path(a.index).read_text())
    rows = source["records"][: a.limit] if a.limit else source["records"]
    if a.describe:
        print(
            json.dumps(
                dict(
                    status="input_spine",
                    scenes=len(rows),
                    source=a.index,
                    frame_policy="previous_and_current_observed",
                    nominal_frame_offsets=[-0.5, 0.0],
                    output=a.output,
                )
            )
        )
        return
    groups = defaultdict(list)
    for row in rows:
        groups[row["log_name"]].append(row)
    manifest = {}
    errors = []
    jobs = [(log, group, a.logs, a.images) for log, group in groups.items()]
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for number, (found, failed) in enumerate(pool.map(read_log, jobs)):
            manifest.update(found)
            errors.extend(failed)
            if number % 20 == 0:
                print(
                    f"logs={number+1}/{len(groups)} scenes={len(manifest)} errors={len(errors)}",
                    flush=True,
                )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        out.with_suffix(".errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError(
            f"{len(errors)} scenes missing causal history; no silent exclusion"
        )
    if len(manifest) != len(rows):
        raise ValueError("Duplicate/missing tokens")
    payload = dict(
        source_index=str(Path(a.index).resolve()),
        frame_policy="previous_and_current_observed",
        nominal_frame_offsets=[-0.5, 0.0],
        crop_top_bottom=28,
        resize_wh=[512, 256],
        records=manifest,
    )
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(out)
    counts = [len(r["auxiliary_indices"]) for r in rows]
    summary = dict(
        scenes=len(rows),
        train=sum(r["split"] == "train" for r in rows),
        val=sum(r["split"] == "val" for r in rows),
        empty_target_scenes=sum(n == 0 for n in counts),
        target_total=sum(counts),
        target_counts={str(n): counts.count(n) for n in sorted(set(counts))},
        frame_policy="previous_and_current_observed",
        teacher_required=False,
        vlm_required=False,
        non_nominal_dt=sum(
            abs(h["offsets"][0] + 0.5) > 0.06 for h in manifest.values()
        ),
        max_dt=max(-h["offsets"][0] for h in manifest.values()),
    )
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
