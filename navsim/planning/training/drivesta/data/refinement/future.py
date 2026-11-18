"""Recorded 4-history/10-future video; auxiliary labels never enter planner."""

import hashlib
from navsim.common.release_paths import metadata_json as json
from pathlib import Path
import numpy as np
import torch

FRAME_POLICY = "drive_jepa_expert_video8_full14_v1"


def assemble_video(row):
    clips = []
    frames = row["frames"]
    if len(frames) != 14:
        raise ValueError("Need actual four history and ten future frame records")
    for start in range(7):
        selected = frames[start : start + 8]
        offsets = [f["offset"] for f in selected]
        paths = [f["image_path"] for f in selected]
        if (
            all(f["available"] for f in selected)
            and len(set(paths)) == 8
            and np.isfinite(offsets).all()
            and offsets[-1] > 0
            and np.all(np.abs(np.diff(offsets) - 0.5) <= 0.100001)
        ):
            clips.append(dict(start=start, offsets=offsets, image_paths=paths))
    return dict(row, video_valid=bool(clips), video_clips=clips)


def load_future_index(path, records, index_identity, history):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if (
        data.get("frame_policy") != FRAME_POLICY
        or data.get("source_identity") != index_identity
    ):
        raise ValueError("Future manifest source/protocol mismatch")
    rows = data["records"]
    for record in records:
        token = record["token"]
        row = rows[token]
        if (
            row["log_name"] != record["log_name"]
            or row["history_paths"] != history[token]["image_paths"]
            or not np.allclose(
                row["history_offsets"], history[token]["offsets"], atol=1e-6, rtol=0
            )
            or row["frames"][3]["token"] != token
            or row["frames"][3]["offset"] != 0
            or not np.allclose(
                [f["offset"] for f in row["frames"][2:4]],
                row["history_offsets"],
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError("Future manifest history/log/timestamp mismatch: " + token)
        expected = assemble_video(row)
        if (
            row["video_valid"] != expected["video_valid"]
            or row["video_clips"] != expected["video_clips"]
        ):
            raise ValueError("Misaligned video manifest: " + token)
    return rows, hashlib.sha256(raw).hexdigest()


def future_target(row, token, epoch, seed=2):
    mask_seed = int.from_bytes(
        hashlib.sha256(f"{token}:{epoch}:{seed}".encode()).digest()[:4], "little"
    ) % (2**31 - 3)
    result = dict(
        jepa_video_clip=torch.zeros(3, 8, 256, 512),
        jepa_video_offsets=torch.zeros(8),
        jepa_future_valid=torch.tensor(row["video_valid"]),
        jepa_mask_seed=torch.tensor(mask_seed),
        jepa_video_start=torch.tensor(-1),
    )
    if not row["video_valid"]:
        return result
    start = int.from_bytes(
        hashlib.sha256(f"{token}:{seed}".encode()).digest()[:8], "little"
    )
    clip = row["video_clips"][(start + epoch) % len(row["video_clips"])]
    import cv2
    from navsim.agents.drivesta.encoders.jepa import preprocess_front

    images = []
    for path in clip["image_paths"]:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result.update(
        jepa_video_clip=preprocess_front(images),
        jepa_video_offsets=torch.tensor(clip["offsets"], dtype=torch.float32),
        jepa_video_start=torch.tensor(clip["start"]),
    )
    return result
