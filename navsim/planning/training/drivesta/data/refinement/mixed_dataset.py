"""Real+SimScale data interface; unchanged V4 network and video objective."""

import hashlib
from navsim.common.release_paths import metadata_json as json, metadata_file_sha256
import pickle
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from navsim.planning.training.drivesta.data.refinement.dataset import DriveSTADataset
from navsim.planning.training.drivesta.data.refinement.future import future_target
from navsim.evaluate.drivesta.refinement.online import OnlinePDM
from navsim.planning.training.drivesta.data.refinement.sampling import cyclic_indices, rotating_replay_indices

VIDEO_POLICY = "drive_jepa_simscale_video8_full12_v1"


def digest(path):
    return metadata_file_sha256(path)


@lru_cache(maxsize=2)
def read_log(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def scene_frames(row):
    frames = read_log(row["log_path"])
    found = [i for i, f in enumerate(frames) if f["token"] == row["token"]]
    if len(found) != 1 or found[0] < 3 or found[0] + 8 >= len(frames):
        raise ValueError("Missing unique4+8 frame record " + row["token"])
    i = found[0]
    if (
        frames[i]["timestamp"] != row["timestamp_us"]
        or frames[i]["log_name"] != row["log_name"]
    ):
        raise ValueError("Synthetic timestamp/log mismatch")
    return frames[i - 3 : i + 9]


def assemble_synthetic_video(row):
    frames = row["frames"]
    if len(frames) != 12 or frames[3]["offset"] != 0:
        raise ValueError("Expected four historical and eight future frames")
    clips = []
    for start in range(5):
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


def synthetic_video_record(row):
    frames = scene_frames(row)
    root = Path(row["sensor_root"])
    sequence = []
    directories = {}
    for f in frames:
        path = (root / f["cams"]["CAM_F0"]["data_path"]).resolve()
        if path.parent not in directories:
            directories[path.parent] = (
                {p.name for p in path.parent.iterdir()}
                if path.parent.is_dir()
                else set()
            )
        sequence.append(
            dict(
                token=f["token"],
                offset=(f["timestamp"] - row["timestamp_us"]) / 1e6,
                image_path=str(path),
                available=path.name in directories[path.parent],
            )
        )
    if not all(f["available"] for f in sequence[2:4]) or sequence[2]["offset"] >= 0:
        raise ValueError("Missing causal JEPA input " + row["token"])
    for cam in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
        if not (root / frames[3]["cams"][cam]["data_path"]).is_file():
            raise FileNotFoundError(row["token"] + " " + cam)
    if not Path(row["official_cache_path"]).is_file():
        raise FileNotFoundError(row["official_cache_path"])
    return row["token"], assemble_synthetic_video(
        dict(
            log_name=row["log_name"],
            history_paths=[f["image_path"] for f in sequence[2:4]],
            history_offsets=[f["offset"] for f in sequence[2:4]],
            frames=sequence,
        )
    )


class SyntheticV4Dataset(Dataset):
    def __init__(self, spec, config, video_path, seed=2):
        self.root = Path(spec["simscale_root"])
        self.config = config
        self.seed = int(seed)
        with open(spec["simscale_manifest"]) as stream:
            self.records = [
                r
                for r in map(json.loads, stream)
                if r["source_group"] in spec["source_groups"]
            ]
        raw = Path(video_path).read_bytes()
        video = json.loads(raw)
        self.future_identity = hashlib.sha256(raw).hexdigest()
        if (
            video["frame_policy"] != VIDEO_POLICY
            or video["source_manifest_sha256"] != spec["simscale_manifest_sha256"]
            or video["source_groups"] != spec["source_groups"]
        ):
            raise ValueError("Synthetic video/source mismatch")
        self.future = video["records"]
        self.future_valid_scenes = 0
        if set(self.future) != {r["token"] for r in self.records}:
            raise ValueError("Video token coverage mismatch")
        for r in self.records:
            v = self.future[r["token"]]
            if v["log_name"] != r["log_name"] or v["frames"][3]["token"] != r["token"]:
                raise ValueError("Video log/token mismatch")
            expected = assemble_synthetic_video(v)
            if (
                v["video_valid"] != expected["video_valid"]
                or v["video_clips"] != expected["video_clips"]
            ):
                raise ValueError("Invalid video window")
            if v["history_paths"] != [f["image_path"] for f in v["frames"][2:4]] or v[
                "history_offsets"
            ] != [f["offset"] for f in v["frames"][2:4]]:
                raise ValueError("Synthetic history mismatch")
            self.future_valid_scenes += int(v["video_valid"])
        if not self.future_valid_scenes:
            raise ValueError("No valid synthetic video supervision")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        from navsim.common.dataclasses import AgentInput, SensorConfig
        from navsim.agents.drivesta.builders.feature_builder import (
            DriveSTAFeatureBuilder,
        )
        from navsim.agents.drivesta.encoders.jepa import preprocess_front
        from pyquaternion import Quaternion
        import cv2

        idx, epoch = item if isinstance(item, tuple) else (item, 0)
        if epoch < 0:
            raise ValueError("Invalid epoch")
        r = self.records[idx]
        token = r["token"]
        folder = self.root / "scenes" / r["split"] / token
        frames = scene_frames(r)
        now = frames[3]
        pose = [
            *now["ego2global_translation"][:2],
            Quaternion(*now["ego2global_rotation"]).yaw_pitch_roll[0],
        ]
        if not np.allclose(pose, r["ego_pose"], atol=1e-5, rtol=0):
            raise ValueError("Synthetic pose mismatch")
        sensor = SensorConfig(
            **{
                k: list(self.config[k])
                for k in (
                    "cam_f0",
                    "cam_l0",
                    "cam_l1",
                    "cam_l2",
                    "cam_r0",
                    "cam_r1",
                    "cam_r2",
                    "cam_b0",
                    "lidar_pc",
                )
            }
        )
        inputs = AgentInput.from_scene_dict_list(
            frames[:4], Path(r["sensor_root"]), 4, sensor
        )
        features = DriveSTAFeatureBuilder(self.config).compute_features(inputs)
        features["scenario_token"] = token
        images = []
        for f in frames[2:4]:
            path = Path(r["sensor_root"]) / f["cams"]["CAM_F0"]["data_path"]
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        features["jepa_clip"] = preprocess_front(images)
        expert = np.asarray(
            json.loads((folder / "expert.json").read_text())["trajectory"], np.float32
        )
        if expert.shape != (8, 3) or not np.isfinite(expert).all():
            raise ValueError("Invalid expert")
        targets = dict(
            trajectory=torch.from_numpy(expert.copy()),
            trajectory_long=torch.from_numpy(expert.copy()),
            trajectory_long_valid=torch.tensor(False),
            token=token,
        )
        full = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
        pool = np.asarray(r["auxiliary_indices"][:32], np.int64)
        if (
            full.shape != (3072,)
            or len(set(pool)) != len(pool)
            or (pool < 0).any()
            or (pool >= 3072).any()
        ):
            raise ValueError("Bad dense bank")
        if not np.array_equal(
            np.asarray(r["auxiliary_labels"][:32]).reshape(-1, 7), full["labels"][pool]
        ):
            raise ValueError("Changed labels")
        key = format(
            int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little"),
            "016x",
        )
        ar = cyclic_indices(pool, 8, epoch, key, self.seed)
        aux = cyclic_indices(pool, 32, epoch, key, self.seed)
        replay = rotating_replay_indices(full["labels"], epoch, key, self.seed)
        for prefix, ids, size in [
            ("auxiliary", aux, 32),
            ("ar_auxiliary", ar, 8),
            ("replay", replay, 64),
        ]:
            paths = np.zeros((size, 8, 3), np.float32)
            paths[: len(ids)] = full["trajectories"][ids]
            targets[prefix + "_trajectories"] = torch.from_numpy(paths)
            targets[prefix + "_mask"] = torch.arange(size) < len(ids)
            targets[prefix + "_query_indices"] = torch.tensor(
                np.pad(ids, (0, size - len(ids)), constant_values=-1)
            )
            if prefix != "ar_auxiliary":
                labels = np.zeros((size, 7), np.float32)
                labels[: len(ids)] = full["labels"][ids]
                targets[prefix + "_labels"] = torch.from_numpy(labels)
        targets.update(
            sampling_epoch=torch.tensor(epoch),
            auxiliary_bank_size=torch.tensor(len(pool)),
        )
        targets.update(future_target(self.future[token], token, epoch, self.seed))
        return features, targets


class DriveSTAMixedDataset(Dataset):
    def __init__(
        self,
        index_path,
        config,
        history_index,
        future_index,
        synthetic_video_index,
        seed=2,
    ):
        raw = Path(index_path).read_bytes()
        self.spec = json.loads(raw)
        s = self.spec
        if s["format"] != "joint_feedback_v2_scaled_v1" or s["source_groups"] != [
            0,
            1,
            2,
        ]:
            raise ValueError("Wrong scaled dataset")
        for k in ("navsim_index", "simscale_manifest", "excluded_tokens"):
            if digest(s[k]) != s[k + "_sha256"]:
                raise ValueError("Changed source " + k)
        self.real = DriveSTADataset(
            s["navsim_index"],
            "trainval",
            seed=seed,
            history_index=history_index,
            future_index=future_index,
        )
        self.synthetic = SyntheticV4Dataset(s, config, synthetic_video_index, seed)
        self.records = [
            dict(r, domain="navsim") for r in self.real.records
        ] + self.synthetic.records
        tokens = [r["token"] for r in self.records]
        if (len(self.real), len(self.synthetic), len(set(tokens))) != (
            103263,
            129409,
            232672,
        ) or s["data_size"] != len(tokens):
            raise ValueError("Wrong DATA_SIZE")
        if set(Path(s["excluded_tokens"]).read_text().splitlines()) & set(tokens):
            raise ValueError("Excluded token in training")
        self.index_identity = hashlib.sha256(raw).hexdigest()
        self.history_identity = self.real.history_identity
        self.future_identity = self.real.future_identity
        self.synthetic_future_identity = self.synthetic.future_identity
        self.future_valid_scenes = (
            self.real.future_valid_scenes + self.synthetic.future_valid_scenes
        )
        self.target_policy = json.loads(Path(s["navsim_index"]).read_text())[
            "target_policy"
        ]
        self.official_cache_root = s["navsim_official_cache_root"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        idx, epoch = item if isinstance(item, tuple) else (item, 0)
        if idx < len(self.real):
            f, t = self.real[(idx, epoch)]
            t["trajectory_long_valid"] = torch.tensor(True)
            return f, t
        return self.synthetic[(idx - len(self.real), epoch)]


class DriveSTAMixedOnlinePDM(OnlinePDM):
    def __init__(self, cache_root, records, workers=4, chunk_size=32):
        super().__init__(cache_root, records, workers, chunk_size)
        for r in records:
            if r.get("domain") == "simscale":
                self.paths[r["token"]] = r["official_cache_path"]
