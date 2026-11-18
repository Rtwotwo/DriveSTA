from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import pickle
import gzip
import os

import torch
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VlmCacheSpec:
    cache_root: Path
    feature_name: str
    hidden_key: str


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


def _config_bool(config: Any, key: str, default: bool = False) -> bool:
    value = getattr(config, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _find_vlm_cache_spec(feature_builders: List[AbstractFeatureBuilder]) -> Optional[VlmCacheSpec]:
    for builder in feature_builders:
        config = getattr(builder, "_config", None)
        if config is None or not _config_bool(config, "use_vlm_feature_cache", False):
            continue

        cache_root = getattr(config, "vlm_feature_cache_path", "")
        if not cache_root:
            raise ValueError(
                "agent.config.vlm_feature_cache_path must be set when " "use_vlm_feature_cache=True"
            )

        feature_name = getattr(config, "vlm_feature_cache_name", "internvl_feature")
        hidden_key = getattr(config, "vlm_hidden_key", "last_hidden_state")
        cache_format = str(getattr(config, "vlm_feature_cache_format", "gz")).lower()
        if cache_format != "gz":
            raise ValueError(
                f"Unsupported vlm_feature_cache_format: {cache_format}. Expected 'gz'."
            )
        return VlmCacheSpec(
            cache_root=Path(str(cache_root)),
            feature_name=str(feature_name),
            hidden_key=str(hidden_key),
        )

    return None


def _to_float_tensor(value: Any) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.detach().float()


def _normalise_vlm_hidden_state(hidden_state: Any) -> torch.Tensor:
    hidden_state_tensor = _to_float_tensor(hidden_state)
    if hidden_state_tensor.ndim == 1:
        hidden_state_tensor = hidden_state_tensor.unsqueeze(0)
    return hidden_state_tensor


def _load_gz_vlm_cache(
    spec: VlmCacheSpec,
    log_name: str,
    token: str,
) -> Dict[str, torch.Tensor]:
    cache_file = spec.cache_root / log_name / token / f"{spec.feature_name}.gz"
    if not cache_file.is_file():
        raise FileNotFoundError(
            f"VLM cache file not found for sample (log={log_name}, token={token}): {cache_file}"
        )

    with gzip.open(cache_file, "rb") as f:
        cache_data = pickle.load(f)

    if spec.hidden_key not in cache_data:
        raise KeyError(
            f"Key '{spec.hidden_key}' not found in VLM cache {cache_file}. "
            f"Available keys: {list(cache_data.keys())}"
        )

    return {"vlm_hidden_state": _normalise_vlm_hidden_state(cache_data[spec.hidden_key])}


def _load_vlm_cache_features(
    spec: VlmCacheSpec,
    log_name: str,
    token: str,
) -> Dict[str, torch.Tensor]:
    return _load_gz_vlm_cache(spec, log_name, token)


def _maybe_add_vlm_cache_features(
    features: Dict[str, torch.Tensor],
    vlm_cache_spec: Optional[VlmCacheSpec],
    log_name: str,
    token: str,
) -> None:
    if vlm_cache_spec is None:
        return

    cache_features = _load_vlm_cache_features(vlm_cache_spec, log_name, token)
    for key, value in cache_features.items():
        if key not in features:
            features[key] = value


def _future_online_offsets(
    feature_builders: List[AbstractFeatureBuilder],
) -> Optional[Tuple[int, ...]]:
    for builder in feature_builders:
        config = getattr(builder, "_config", None)
        if config is None or not _config_bool(config, "use_future_online_internvl", False):
            continue
        offsets = tuple(int(value) for value in getattr(config, "future_image_offsets", (2, 5, 8)))
        if len(offsets) != 3 or len(set(offsets)) != 3 or min(offsets) < 1:
            raise ValueError("Online future InternVL requires three distinct positive offsets")
        return offsets
    return None


def load_future_internvl_pixels(scene: Any, offsets: Tuple[int, ...]) -> torch.Tensor:
    """Read privileged recorded front-camera frames for the online teacher."""
    from PIL import Image
    import numpy as np
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    # Keep the generic dataset independent of concrete Agent implementations.
    # These operations match the teacher's InternVL 448px preprocessing exactly.
    transform = T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    current_index = scene.scene_metadata.num_history_frames - 1
    images = []
    for offset in offsets:
        index = current_index + offset
        if index >= len(scene.frames):
            raise IndexError(f"Future image offset {offset} exceeds scene window")
        value = scene.frames[index].cameras.cam_f0.image
        if isinstance(value, np.ndarray):
            image = Image.fromarray(value).convert("RGB")
        elif isinstance(value, (str, Path)):
            with Image.open(value) as opened:
                image = opened.convert("RGB")
        else:
            raise ValueError(f"CAM_F0 missing at future offset {offset}")
        images.append(transform(image))
    return torch.stack(images)


@dataclass(frozen=True)
class FutureLatentCacheSpec:
    """Locates the compact Wan-VAE future-latent distillation cache."""

    cache_root: Path
    feature_name: str
    latent_shape: Tuple[int, ...]
    key: str


def _find_future_latent_cache_spec(
    feature_builders: List[AbstractFeatureBuilder],
) -> Optional[FutureLatentCacheSpec]:
    for builder in feature_builders:
        config = getattr(builder, "_config", None)
        if config is None or not _config_bool(config, "use_cosmos_future_cache", False):
            continue
        cache_root = getattr(config, "cosmos_future_cache_path", "")
        if not cache_root:
            raise ValueError(
                "agent.config.cosmos_future_cache_path must be set when "
                "use_cosmos_future_cache=True"
            )
        future_tokens = getattr(config, "world_future_tokens", None)
        future_dim = getattr(config, "world_future_dim", None)
        if future_tokens is not None and future_dim is not None:
            latent_shape = (int(future_tokens), int(future_dim))
        else:
            latent_shape = (
                int(getattr(config, "world_future_channels", 48)),
                int(getattr(config, "world_future_time", 2)),
                int(getattr(config, "world_future_height", 4)),
                int(getattr(config, "world_future_width", 8)),
            )
        return FutureLatentCacheSpec(
            cache_root=Path(str(cache_root)),
            feature_name=str(getattr(config, "cosmos_future_cache_name", "cosmos_future_latent")),
            latent_shape=latent_shape,
            key=str(getattr(config, "cosmos_future_cache_key", "future_latent")),
        )
    return None


def _maybe_add_future_latent_target(
    targets: Dict[str, torch.Tensor],
    spec: Optional[FutureLatentCacheSpec],
    log_name: str,
    token: str,
) -> None:
    """Inject the compact future latent as a privileged training target.

    Missing or unreadable caches yield a zero target flagged invalid via
    ``cosmos_future_valid`` so the world loss can skip them without bias.
    """
    if spec is None:
        return
    cache_file = spec.cache_root / log_name / token / f"{spec.feature_name}.gz"
    if cache_file.is_file():
        try:
            with gzip.open(cache_file, "rb") as handle:
                cache_data = pickle.load(handle)
            latent = cache_data[spec.key]
            latent = latent if torch.is_tensor(latent) else torch.as_tensor(latent)
            targets["cosmos_future_latent"] = latent.detach().float()
            targets["cosmos_future_valid"] = torch.ones((), dtype=torch.bool)
            return
        except (OSError, EOFError, KeyError, pickle.UnpicklingError) as exc:
            logger.warning("Failed to load future latent cache %s: %s", cache_file, exc)
    targets["cosmos_future_latent"] = torch.zeros(spec.latent_shape, dtype=torch.float32)
    targets["cosmos_future_valid"] = torch.zeros((), dtype=torch.bool)


def _load_builder_cache(
    builders: List[AbstractFeatureBuilder],
    token_path: Path,
) -> Dict[str, torch.Tensor]:
    data: Dict[str, torch.Tensor] = {}
    for builder in builders:
        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
        data.update(load_feature_target_from_pickle(data_dict_path))
    return data


def _load_target_cache(
    builders: List[AbstractTargetBuilder],
    token_path: Path,
) -> Dict[str, torch.Tensor]:
    data: Dict[str, torch.Tensor] = {}
    for builder in builders:
        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
        data.update(load_feature_target_from_pickle(data_dict_path))
    return data


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = sorted(
                (
                    Path(log_name)
                    for log_name in log_names
                    if (self._cache_path / log_name).is_dir()
                ),
                key=lambda path: path.name,
            )
        else:
            self.log_names = sorted(
                (log_name for log_name in self._cache_path.iterdir() if log_name.is_dir()),
                key=lambda path: path.name,
            )

        self._feature_builders = feature_builders
        if any(
            bool(getattr(builder, "requires_scene_features", False)) for builder in feature_builders
        ):
            raise ValueError("Scene-aware online builders cannot use CacheOnlyDataset")
        self._target_builders = target_builders
        self._vlm_cache_spec = _find_vlm_cache_spec(self._feature_builders)
        self._future_latent_spec = _find_future_latent_cache_spec(self._feature_builders)
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self.tokens = sorted(self._valid_cache_paths)

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in sorted(
                (path for path in log_path.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            ):
                found_caches = [
                    (token_path / (builder.get_unique_name() + ".gz")).is_file()
                    for builder in feature_builders + target_builders
                ]
                if not all(found_caches):
                    continue
                if token_path.name in valid_cache_paths:
                    raise RuntimeError(f"Duplicate cache token across logs: {token_path.name}")
                valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(
        self, token: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features = _load_builder_cache(self._feature_builders, token_path)
        _maybe_add_vlm_cache_features(
            features=features,
            vlm_cache_spec=self._vlm_cache_spec,
            log_name=token_path.parent.name,
            token=token,
        )

        targets = _load_target_cache(self._target_builders, token_path)
        _maybe_add_future_latent_target(
            targets=targets,
            spec=self._future_latent_spec,
            log_name=token_path.parent.name,
            token=token,
        )
        return features, targets


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        append_token_to_batch: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._vlm_cache_spec = _find_vlm_cache_spec(self._feature_builders)
        self._future_latent_spec = _find_future_latent_cache_spec(self._feature_builders)
        self._scene_feature_builders = [
            builder
            for builder in self._feature_builders
            if bool(getattr(builder, "requires_scene_features", False))
        ]
        if self._scene_feature_builders and cache_path is not None:
            raise ValueError("Scene-aware online builders require cache_path=null")
        self._future_online_offsets = _future_online_offsets(self._feature_builders)
        if self._future_online_offsets is not None and cache_path is not None:
            raise ValueError(
                "Online future InternVL requires cache_path=null to read recorded images"
            )

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )
        self.append_token_to_batch = append_token_to_batch
        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(
        self, token: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features = _load_builder_cache(self._feature_builders, token_path)
        _maybe_add_vlm_cache_features(
            features=features,
            vlm_cache_spec=self._vlm_cache_spec,
            log_name=token_path.parent.name,
            token=token,
        )

        targets = _load_target_cache(self._target_builders, token_path)
        _maybe_add_future_latent_target(
            targets=targets,
            spec=self._future_latent_spec,
            log_name=token_path.parent.name,
            token=token,
        )
        return features, targets

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._scene_feature_builders:
                features.update(builder.compute_scene_features(scene))
            if self._future_online_offsets is not None:
                features["future_pixel_values"] = load_future_internvl_pixels(
                    scene, self._future_online_offsets
                )
            _maybe_add_vlm_cache_features(
                features=features,
                vlm_cache_spec=self._vlm_cache_spec,
                log_name=scene.scene_metadata.log_name,
                token=scene.scene_metadata.initial_token,
            )
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))
            _maybe_add_future_latent_target(
                targets=targets,
                spec=self._future_latent_spec,
                log_name=scene.scene_metadata.log_name,
                token=scene.scene_metadata.initial_token,
            )

        features["scenario_token"] = token
        if self.append_token_to_batch:
            return features, targets, token
        return features, targets
