"""Split-separated contiguous training shards and indexed zero-copy scene access."""

from pathlib import Path
import os
import numpy as np
from .store import atomic, checksum, digest, load_blocks, read_json, write_json, lock

ROW_DTYPE = np.dtype(
    [
        ("keys", "<i4", (2,)),
        ("trajectories", "<f4", (8, 3)),
        ("labels", "<f4", (7,)),
        ("phase", "<i2"),
    ],
    align=True,
)


def pack_dataset(run, output, max_bytes=1024 * 1024 * 1024):
    run, output = Path(run), Path(output)
    completed = read_json(run / "COMPLETE.json")
    audit = read_json(run / "audit.json")
    if not audit["passed"] or not all(a["official"] for a in audit["scenes"]):
        raise ValueError("Official audit required before packing")
    scenes = [run / "scenes" / r["split"] / r["token"] for r in completed["scenes"]]
    groups = []
    group = []
    size = 0
    split = None
    for scene in sorted(scenes):
        meta = read_json(scene / "COMPLETE.json")
        for phase in (scene / "phases").iterdir():
            for block in read_json(phase / "COMPLETE.json")["blocks"]:
                if checksum(phase / block["file"]) != block["sha256"]:
                    raise ValueError("Source block changed after audit")
        n = meta["candidates"] * ROW_DTYPE.itemsize
        if group and (size + n > max_bytes or split != meta["split"]):
            groups.append((split, group))
            group = []
            size = 0
        split = meta["split"]
        group.append(scene)
        size += n
    if group:
        groups.append((split, group))
    index = []
    shards = []
    with lock(output):
        contract = dict(
            run_contract=read_json(run / "run.json"),
            scene_counts=[
                (str(s.relative_to(run)), read_json(s / "COMPLETE.json")["candidates"])
                for s in scenes
            ],
            row_dtype=ROW_DTYPE.descr,
            shard_max_bytes=max_bytes,
        )
        if (output / "contract.json").exists() and digest(
            read_json(output / "contract.json")
        ) != digest(contract):
            raise ValueError("Packing output has another source/configuration")
        write_json(output / "contract.json", contract)
        for shard_id, (split, group) in enumerate(groups):
            target = output / f"{split}-{shard_id:05d}.bin"
            shard_index = []
            offset = 0
            for scene in group:
                expert = read_json(scene / "expert.json")
                meta = read_json(scene / "COMPLETE.json")
                shard_index.append(
                    dict(
                        token=meta["token"],
                        split=split,
                        log_name=meta["log_name"],
                        shard=target.name,
                        row_offset=offset,
                        count=meta["candidates"],
                        expert=expert,
                        phases=[p["name"] for p in meta["phases"]],
                        coverage=meta["coverage"],
                    )
                )
                offset += meta["candidates"]

            def writer(f):
                for scene in group:
                    phase_names = [p["name"] for p in read_json(scene / "COMPLETE.json")["phases"]]
                    for path, z in load_blocks(scene):
                        # Aligned rows give float fields strides divisible by four for tensor views.
                        # Zero initialization makes the two padding bytes deterministic.
                        data = np.zeros(len(z["keys"]), dtype=ROW_DTYPE)
                        for key in ("keys", "trajectories", "labels"):
                            data[key] = z[key]
                        data["phase"] = phase_names.index(path.parent.name)
                        if np.max(abs(data["labels"].astype(np.float64) - z["labels"])) > 1e-6:
                            raise ValueError("Float32 label precision exceeded")
                        f.write(data.tobytes())

            def validate(path):
                if path.stat().st_size != offset * ROW_DTYPE.itemsize:
                    raise ValueError("Packed shard length mismatch")
                mmap = np.memmap(path, dtype=ROW_DTYPE, mode="r")
                cursor = 0
                for scene in group:
                    for _, z in load_blocks(scene):
                        segment = mmap[cursor : cursor + len(z["keys"])]
                        for key in ("keys", "trajectories", "labels"):
                            if not np.array_equal(segment[key], z[key].astype(segment[key].dtype)):
                                raise ValueError("Packed rows differ from source blocks")
                        cursor += len(z["keys"])
                del mmap

            if target.exists():
                validate(target)
            else:
                atomic(target, writer, validate)
            index.extend(shard_index)
            shards.append(
                dict(
                    file=target.name,
                    split=split,
                    rows=offset,
                    bytes=target.stat().st_size,
                    sha256=checksum(target),
                )
            )
            write_json(target.with_suffix(".json"), dict(shard=shards[-1], scenes=shard_index))
        write_json(
            output / "index.json",
            dict(schema_version=1, row_dtype=ROW_DTYPE.descr, scenes=index, shards=shards),
        )
        result = dict(
            complete=True,
            scenes=len(index),
            candidates=sum(s["rows"] for s in shards),
            shards=len(shards),
            bytes=sum(s["bytes"] for s in shards),
            splits=sorted(set(s["split"] for s in index)),
        )
        write_json(output / "COMPLETE.json", result)
    return result


class PackedDataset:
    """One index per scene; callers can sample scenes uniformly before candidates."""

    def __init__(self, directory, split):
        self.root = Path(directory)
        if not (self.root / "COMPLETE.json").exists():
            raise ValueError("Packing incomplete")
        index = read_json(self.root / "index.json")
        self.scenes = [s for s in index["scenes"] if s["split"] == split]
        self._maps = {}

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        info = self.scenes[index]
        name = info["shard"]
        if name not in self._maps:
            self._maps[name] = np.memmap(self.root / name, dtype=ROW_DTYPE, mode="r")
        return info, self._maps[name][info["row_offset"] : info["row_offset"] + info["count"]]
