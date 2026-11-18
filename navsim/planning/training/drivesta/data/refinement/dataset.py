"""Read expert targets, screened auxiliaries, and stratified scored replay."""

from navsim.common.compatibility.checkpoints import (
    cached_feature_path,
    cached_target_path,
)

import gzip
import hashlib
from navsim.common.release_paths import metadata_json as json, metadata_sha256
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from navsim.planning.training.drivesta.data.refinement.sampling import cyclic_indices, rotating_replay_indices


def read_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


class DriveSTADataset(Dataset):
    def __init__(
        self,
        index_path,
        split,
        use_vlm=False,
        replay_samples=64,
        limit=None,
        seed=2,
        auxiliary_samples=32,
        auxiliary_bank_capacity=32,
        history_index=None,
        future_index=None,
    ):
        if split not in ("train", "val", "trainval") or replay_samples != 64:
            raise ValueError("Invalid split/replay size")
        raw = Path(index_path).read_bytes()
        self.index_identity = metadata_sha256(raw)
        index = json.loads(raw)
        if not index.get("target_policy", {}).get("check_planned_space"):
            raise ValueError("Screened spatial target index required")
        if auxiliary_samples != 32 or auxiliary_bank_capacity != 32:
            raise ValueError("Refinement reads all up-to-32 screened representatives")
        if index["target_policy"].get("max_targets", 0) > auxiliary_bank_capacity:
            raise ValueError("Target index exceeds capacity")
        self.seed = int(seed)
        self.auxiliary_samples = auxiliary_samples
        self.auxiliary_bank_capacity = auxiliary_bank_capacity
        self.records = [
            r for r in index["records"] if split == "trainval" or r["split"] == split
        ]
        if limit is not None:
            self.records = self.records[: int(limit)]
        if use_vlm:
            raise ValueError("Refinement JEPA never reads VLM caches")
        history = json.loads(Path(history_index).read_text())
        if history.get("frame_policy") != "previous_and_current_observed":
            raise ValueError("Wrong causal frame manifest")
        self.history = history["records"]
        self.history_identity = hashlib.sha256(
            Path(history_index).read_bytes()
        ).hexdigest()
        if any(r["token"] not in self.history for r in self.records):
            raise ValueError("Missing history frames")
        self.root = Path(index["root"])
        self.cache_path = Path(index["cache_path"])
        self.vlm_path = Path(index["vlm_path"]) if index["vlm_path"] else None
        self.split = split
        self.use_vlm = use_vlm
        if use_vlm and self.vlm_path is None:
            raise ValueError("Refinement needs VLM cache")
        for r in self.records:
            if "auxiliary_indices" not in r:
                raise ValueError("Unscreened scene " + r["token"])
            h = self.history[r["token"]]
            if (
                h["log_name"] != r["log_name"]
                or not h["offsets"][0] < 0
                or h["offsets"][1] != 0
            ):
                raise ValueError("Causal history/log mismatch " + r["token"])

        from navsim.planning.training.drivesta.data.refinement.future import load_future_index

        self.future, self.future_identity = load_future_index(
            future_index, self.records, self.index_identity, self.history
        )
        self.future_valid_scenes = sum(
            self.future[r["token"]]["video_valid"] for r in self.records
        )
        if self.records and self.future_valid_scenes == 0:
            raise ValueError(
                "No real future supervision in this split; refusing silent frozen-JEPA training"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        idx, epoch = idx if isinstance(idx, tuple) else (idx, 0)
        if epoch < 0:
            raise ValueError("Invalid sampling epoch")
        r = self.records[idx]
        token = r["token"]
        log = r["log_name"]
        folder = self.root / "scenes" / r["split"] / token
        cache = self.cache_path / log / token
        features = read_pickle(cached_feature_path(cache))
        old = read_pickle(cached_target_path(cache))
        if old["token"] != token or "trajectory_long" not in old:
            raise ValueError("Expert cache mismatch " + token)
        expert = np.asarray(
            json.loads((folder / "expert.json").read_text())["trajectory"],
            dtype=np.float32,
        )
        if not np.allclose(np.asarray(old["trajectory"]), expert, atol=1e-4, rtol=0):
            raise ValueError("Expert/cache alignment mismatch")
        targets = {k: old[k] for k in ("trajectory", "trajectory_long", "token")}
        from navsim.agents.drivesta.encoders.jepa import preprocess_front
        import cv2

        images = []
        for path in self.history[token]["image_paths"]:
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(path)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        features["jepa_clip"] = preprocess_front(images)
        features["scenario_token"] = token
        if self.use_vlm:
            features["vlm_hidden_state"] = (
                torch.as_tensor(
                    read_pickle(self.vlm_path / log / token / "internvl_feature.gz")[
                        "last_hidden_state"
                    ]
                )
                .detach()
                .float()
            )
        full = np.load(folder / "candidates.npy", mmap_mode="r", allow_pickle=False)
        aux_idx = np.asarray(r["auxiliary_indices"], dtype=np.int64)
        if (
            len(aux_idx) > self.auxiliary_bank_capacity
            or len(np.unique(aux_idx)) != len(aux_idx)
            or (aux_idx < 0).any()
            or (aux_idx >= len(full)).any()
        ):
            raise ValueError("Invalid auxiliary index")
        if not np.array_equal(
            np.asarray(r["auxiliary_labels"]).reshape(-1, 7), full["labels"][aux_idx]
        ):
            raise ValueError("Source labels changed")
        bank_size = len(aux_idx)
        ar_idx = cyclic_indices(aux_idx, 8, epoch, token, self.seed)
        aux_idx = cyclic_indices(
            aux_idx, self.auxiliary_samples, epoch, token, self.seed
        )
        aux = np.zeros((32, 8, 3), dtype=np.float32)
        aux_labels = np.zeros((32, 7), dtype=np.float32)
        aux_mask = np.zeros(32, dtype=bool)
        aux[: len(aux_idx)] = full["trajectories"][aux_idx]
        aux_labels[: len(aux_idx)] = full["labels"][aux_idx]
        aux_mask[: len(aux_idx)] = True
        ar_aux = np.zeros((8, 8, 3), dtype=np.float32)
        ar_mask = np.zeros(8, dtype=bool)
        ar_aux[: len(ar_idx)] = full["trajectories"][ar_idx]
        ar_mask[: len(ar_idx)] = True
        indices = rotating_replay_indices(full["labels"], epoch, token, self.seed)
        replay = np.zeros((64, 8, 3), dtype=np.float32)
        labels = np.zeros((64, 7), dtype=np.float32)
        mask = np.zeros(64, dtype=bool)
        replay[: len(indices)] = full["trajectories"][indices]
        labels[: len(indices)] = full["labels"][indices]
        mask[: len(indices)] = True
        for k, value in dict(
            auxiliary_trajectories=aux,
            auxiliary_labels=aux_labels,
            auxiliary_mask=aux_mask,
            ar_auxiliary_trajectories=ar_aux,
            ar_auxiliary_mask=ar_mask,
            replay_trajectories=replay,
            replay_labels=labels,
            replay_mask=mask,
        ).items():
            targets[k] = torch.from_numpy(value.copy())
        targets["sampling_epoch"] = torch.tensor(epoch, dtype=torch.int64)
        targets["auxiliary_bank_size"] = torch.tensor(bank_size, dtype=torch.int64)
        targets["auxiliary_query_indices"] = torch.tensor(
            np.pad(aux_idx, (0, 32 - len(aux_idx)), constant_values=-1)
        )
        targets["ar_auxiliary_query_indices"] = torch.tensor(
            np.pad(ar_idx, (0, 8 - len(ar_idx)), constant_values=-1)
        )
        targets["replay_query_indices"] = torch.tensor(
            np.pad(indices, (0, 64 - len(indices)), constant_values=-1)
        )
        from navsim.planning.training.drivesta.data.refinement.future import future_target

        targets.update(future_target(self.future[token], token, epoch, self.seed))
        return features, targets
