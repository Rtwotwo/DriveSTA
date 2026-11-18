"""Index real 4-history/10-future video from the COMPLETE sensor directory."""

import argparse, hashlib, pickle, sys
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from navsim.planning.training.drivesta.data.refinement.future import FRAME_POLICY, assemble_video


import os

ROOT = Path(__file__).resolve().parents[3]
from navsim.common.release_paths import metadata_json as json, metadata_sha256


def read_log(job):
    log, group, logs, images, history = job
    with (Path(logs) / (log + ".pkl")).open("rb") as stream:
        frames = pickle.load(stream)
    lookup = {f["token"]: i for i, f in enumerate(frames)}
    if len(lookup) != len(frames):
        raise ValueError("Duplicate log tokens: " + log)
    manifest = {}
    existence = {}

    def available(path):
        parent = path.parent
        if parent not in existence:
            existence[parent] = (
                {p.name for p in parent.iterdir() if p.is_file()}
                if parent.is_dir()
                else set()
            )
        return path.name in existence[parent]

    for row in group:
        token = row["token"]
        i = lookup[token]
        if i < 3 or i + 10 >= len(frames):
            raise ValueError("Missing raw 4+10 frame records: " + token)
        now = frames[i]["timestamp"]
        h = history[token]
        offsets = [(f["timestamp"] - now) / 1e6 for f in frames[i - 1 : i + 1]]
        if offsets != h["offsets"] or h["log_name"] != log:
            raise ValueError("History mismatch " + token)
        for f, path in zip(frames[i - 1 : i + 1], h["image_paths"]):
            if not str(path).endswith(f["cams"]["CAM_F0"]["data_path"]):
                raise ValueError("History relative path mismatch")
        sequence = []
        for f in frames[i - 3 : i + 11]:
            path = (Path(images) / f["cams"]["CAM_F0"]["data_path"]).resolve()
            sequence.append(
                dict(
                    token=f["token"],
                    offset=(f["timestamp"] - now) / 1e6,
                    image_path=str(path),
                    available=available(path),
                )
            )
        manifest[token] = assemble_video(
            dict(
                log_name=log,
                history_paths=h["image_paths"],
                history_offsets=offsets,
                frames=sequence,
            )
        )
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index",
        default=str(
            ROOT / "data/indexes/drivesta/proposals.json"
        ),
    )
    p.add_argument(
        "--history",
        default=str(
            ROOT
            / "data/indexes/drivesta/history.json"
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
            / "sensor_blobs/trainval_all"
        ),
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "outputs/indexes/drivesta_future/video8_full14_index.json"),
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--describe", action="store_true")
    a = p.parse_args()
    raw = Path(a.index).read_bytes()
    source = json.loads(raw)
    rows = source["records"][: a.limit] if a.limit else source["records"]
    if a.describe:
        print(
            json.dumps(
                dict(
                    scenes=len(rows),
                    frame_policy=FRAME_POLICY,
                    images=str(Path(a.images).resolve()),
                )
            )
        )
        return
    history = json.loads(Path(a.history).read_text())
    if history.get("frame_policy") != "previous_and_current_observed":
        raise ValueError("Wrong history protocol")
    groups = defaultdict(list)
    for row in rows:
        groups[row["log_name"]].append(row)
    jobs = [
        (
            log,
            group,
            a.logs,
            a.images,
            {r["token"]: history["records"][r["token"]] for r in group},
        )
        for log, group in groups.items()
    ]
    records = {}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, found in enumerate(pool.map(read_log, jobs)):
            if records.keys() & found.keys():
                raise ValueError("Duplicate scene token")
            records.update(found)
            if i % 40 == 0:
                print(f"logs={i+1}/{len(jobs)} scenes={len(records)}", flush=True)
    if len(records) != len(rows):
        raise ValueError("Missing scene")
    payload = dict(
        frame_policy=FRAME_POLICY,
        source_identity=metadata_sha256(raw),
        source_index=str(Path(a.index).resolve()),
        sensor_root=str(Path(a.images).resolve()),
        records=records,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out)
    counts = Counter(len(r["video_clips"]) for r in records.values())
    summary = dict(
        scenes=len(records),
        video8_valid=sum(r["video_valid"] for r in records.values()),
        video8_missing=sum(not r["video_valid"] for r in records.values()),
        windows_count=dict(counts),
        all14_images_present=sum(
            all(f["available"] for f in r["frames"]) for r in records.values()
        ),
        sensor_root=str(Path(a.images).resolve()),
        future_images_are_auxiliary_labels_only=True,
    )
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    out.with_suffix(".exceptions.json").write_text(
        json.dumps({t: r for t, r in records.items() if not r["video_valid"]})
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
