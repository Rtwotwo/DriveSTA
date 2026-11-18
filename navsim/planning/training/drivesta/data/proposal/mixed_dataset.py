"""V2 data scaling; original real reader, model and losses retain their semantics."""

import hashlib
from navsim.common.release_paths import metadata_json as json, metadata_file_sha256
import pickle
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from navsim.planning.training.drivesta.data.proposal.dataset import DriveSTAProposalDataset
from navsim.planning.training.drivesta.data.proposal.sampling import cyclic_indices, rotating_replay_indices
from navsim.evaluate.drivesta.proposal.online import OnlinePDM


def digest(path):
    return metadata_file_sha256(path)


def diverse_indices(paths, labels, indices, min_ade=0.25):
    """Descending quality, then stable full-trajectory separation; no repeats."""
    selected = []
    for i in np.argsort(-np.asarray(labels)[:, 6], kind="stable"):
        if selected:
            ade = np.linalg.norm(
                paths[selected, :, :2] - paths[i, :, :2], axis=-1
            ).mean(-1)
            if (ade < min_ade).any():
                continue
        selected.append(int(i))
    return np.asarray(indices)[selected]


@lru_cache(maxsize=2)
def read_log(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def current_history(record):
    frames = read_log(record["log_path"])
    indices = [i for i, f in enumerate(frames) if f["token"] == record["token"]]
    if len(indices) != 1 or indices[0] < 3:
        raise ValueError("Ambiguous/missing history " + record["token"])
    frames = frames[indices[0] - 3 : indices[0] + 1]
    if frames[-1]["log_name"] != record["log_name"]:
        raise ValueError("Synthetic log mismatch")
    if int(frames[-1]["timestamp"]) != record["timestamp_us"]:
        raise ValueError("Synthetic timestamp mismatch")
    return frames


class SimScaleV2Dataset(Dataset):
    def __init__(self, manifest, root, feature_config, groups, seed=2):
        self.root = Path(root)
        self.feature_config = feature_config
        self.seed = int(seed)
        self.records = []
        with Path(manifest).open() as stream:
            for row in map(json.loads, stream):
                if row["source_group"] not in groups:
                    continue
                row["auxiliary_indices"] = row["auxiliary_indices"][:32]
                row["auxiliary_labels"] = row["auxiliary_labels"][:32]
                self.records.append(row)

    def __len__(self):
        return len(self.records)

    def read_inputs(self, r, folder, cache):
        from navsim.common.dataclasses import AgentInput, SensorConfig
        from navsim.agents.drivesta.builders.feature_builder import (
            DriveSTAFeatureBuilder,
        )
        from pyquaternion import Quaternion

        frames = current_history(r)
        f = frames[-1]
        pose = [
            *f["ego2global_translation"][:2],
            Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0],
        ]
        if not np.allclose(pose, r["ego_pose"], atol=1e-5, rtol=0):
            raise ValueError("Synthetic pose mismatch")
        sensor = SensorConfig(
            **{
                name: list(self.feature_config[name])
                for name in (
                    "cam_f0",
                    "cam_b0",
                    "cam_l0",
                    "cam_l1",
                    "cam_l2",
                    "cam_r0",
                    "cam_r1",
                    "cam_r2",
                    "lidar_pc",
                )
            }
        )
        agent_input = AgentInput.from_scene_dict_list(
            frames, Path(r["sensor_root"]), 4, sensor
        )
        features = DriveSTAFeatureBuilder(self.feature_config).compute_features(
            agent_input
        )
        if features["image"].shape[0] != 4:
            raise ValueError("Expected four actual camera observations")
        expert = np.asarray(
            json.loads((folder / "expert.json").read_text())["trajectory"],
            dtype=np.float32,
        )
        if expert.shape != (8, 3) or not np.isfinite(expert).all():
            raise ValueError("Invalid four-second expert")
        trajectory = torch.from_numpy(expert.copy())
        return features, dict(
            trajectory=trajectory,
            trajectory_long=trajectory.clone(),
            trajectory_long_valid=torch.tensor(False),
            token=r["token"],
        )

    def __getitem__(self, item):
        index, epoch = item if isinstance(item, tuple) else (item, 0)
        if epoch < 0:
            raise ValueError("Invalid sampling epoch")
        row = self.records[index]
        token = row["token"]
        folder = self.root / "scenes" / row["split"] / token
        features, targets = self.read_inputs(row, folder, None)
        features["scenario_token"] = token
        full = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
        if full.shape != (3072,):
            raise ValueError("Invalid complete bank " + token)
        pool = np.asarray(row["auxiliary_indices"], dtype=np.int64)
        if (
            len(pool) > 32
            or len(set(pool)) != len(pool)
            or (pool < 0).any()
            or (pool >= 3072).any()
        ):
            raise ValueError("Invalid auxiliary indices " + token)
        if not np.array_equal(
            np.asarray(row["auxiliary_labels"]).reshape(-1, 7), full["labels"][pool]
        ):
            raise ValueError("Published labels changed " + token)
        # Same V2 rotation, with a stable numeric seed for suffixed SimScale tokens.
        key = format(
            int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little"),
            "016x",
        )
        chosen = cyclic_indices(pool, 8, epoch, key, self.seed)
        replay = rotating_replay_indices(full["labels"], epoch, key, self.seed)
        for prefix, ids, size in [("auxiliary", chosen, 8), ("replay", replay, 64)]:
            paths = np.zeros((size, 8, 3), np.float32)
            labels = np.zeros((size, 7), np.float32)
            mask = np.zeros(size, bool)
            paths[: len(ids)] = full["trajectories"][ids]
            labels[: len(ids)] = full["labels"][ids]
            mask[: len(ids)] = True
            targets[prefix + "_trajectories"] = torch.from_numpy(paths)
            targets[prefix + "_labels"] = torch.from_numpy(labels)
            targets[prefix + "_mask"] = torch.from_numpy(mask)
            targets[prefix + "_query_indices"] = torch.tensor(
                np.pad(ids, (0, size - len(ids)), constant_values=-1)
            )
        targets["sampling_epoch"] = torch.tensor(epoch)
        targets["auxiliary_bank_size"] = torch.tensor(len(pool))
        return features, targets


class DriveSTAProposalMixedDataset(Dataset):
    def __init__(self, index_path, feature_config, seed=2):
        raw = Path(index_path).read_bytes()
        spec = json.loads(raw)
        self.spec = spec
        if spec["format"] != "joint_feedback_v2_scaled_v1":
            raise ValueError("Wrong V2 scaling index")
        for key in ("navsim_index", "simscale_manifest", "excluded_tokens"):
            if digest(spec[key]) != spec[key + "_sha256"]:
                raise ValueError("Changed dependency " + key)
        self.real = DriveSTAProposalDataset(spec["navsim_index"], "trainval", seed=seed)
        self.synthetic = SimScaleV2Dataset(
            spec["simscale_manifest"],
            spec["simscale_root"],
            feature_config,
            spec["source_groups"],
            seed,
        )
        if (len(self.real), len(self.synthetic)) != (103263, 129409):
            raise ValueError("Wrong source counts")
        self.records = [
            dict(r, domain="navsim") for r in self.real.records
        ] + self.synthetic.records
        tokens = [r["token"] for r in self.records]
        if len(set(tokens)) != 232672 or len(tokens) != spec["data_size"]:
            raise ValueError("Overlapping/missing token")
        if set(Path(spec["excluded_tokens"]).read_text().splitlines()).intersection(
            tokens
        ):
            raise ValueError("Excluded token entered training")
        self.target_policy = json.loads(Path(spec["navsim_index"]).read_text())[
            "target_policy"
        ]
        self.index_identity = hashlib.sha256(raw).hexdigest()
        self.official_cache_root = spec["navsim_official_cache_root"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        index, epoch = item if isinstance(item, tuple) else (item, 0)
        if index < len(self.real):
            features, targets = self.real[(index, epoch)]
            targets["trajectory_long_valid"] = torch.tensor(True)
        else:
            features, targets = self.synthetic[(index - len(self.real), epoch)]
        return features, targets


class DriveSTAProposalMixedOnlinePDM(OnlinePDM):
    def __init__(self, cache_root, records, workers=4):
        super().__init__(cache_root, records, workers)
        for row in records:
            if row.get("domain") == "simscale":
                # Canonical published actor-tail cache; never original 4-second cache.
                self.paths[row["token"]] = row["official_cache_path"]
