from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


LOWDIM_KEYS = [
    "state",
    "mocap_xy",
    "object_pose",
    "goal_pose",
]

CAM_TOP_KEY = "observation.images.cam_top"
CAM_SIDE_KEY = "observation.images.cam_side"
CACHE_VERSION = 1


def to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def image_to_chw_float(x: Any) -> torch.Tensor:
    x = to_tensor(x)

    if x.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {tuple(x.shape)}")

    # LeRobot / video backends may return CHW or HWC depending on version.
    if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        chw = x
    else:
        chw = x.permute(2, 0, 1)

    chw = chw.to(dtype=torch.float32).contiguous()
    if chw.max().item() > 2.0:
        chw = chw / 255.0
    return chw


def resolve_sample_key(sample: dict[str, Any], key: str) -> str:
    if key in sample:
        return key

    observation_key = f"observation.{key}"
    if observation_key in sample:
        return observation_key

    if key.startswith("observation."):
        short_key = key.removeprefix("observation.")
        if short_key in sample:
            return short_key

    raise KeyError(f"missing key: {key}")


def lowdim_from_sample(sample: dict[str, Any]) -> torch.Tensor:
    pieces = [
        to_tensor(sample[resolve_sample_key(sample, key)]).reshape(-1)
        for key in LOWDIM_KEYS
    ]
    return torch.cat(pieces, dim=0)


def action_from_sample(sample: dict[str, Any]) -> torch.Tensor:
    if "action" not in sample:
        raise KeyError("missing key: action")
    return to_tensor(sample["action"]).reshape(-1)


def make_indices(total: int, max_items: int | None) -> list[int]:
    if max_items is None:
        return list(range(total))
    return list(range(min(max_items, total)))


@dataclass(frozen=True)
class LowdimCache:
    values: torch.Tensor | None = None
    actions: torch.Tensor | None = None


@dataclass(frozen=True)
class ImageCache:
    cam_top: torch.Tensor | None = None
    cam_side: torch.Tensor | None = None


class CacheMixin:
    dataset_dir: Path
    indices: list[int]

    def _metadata(self, name: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": CACHE_VERSION,
            "name": name,
            "dataset_dir": str(self.dataset_dir),
            "indices": {
                "count": len(self.indices),
                "first": self.indices[0] if self.indices else None,
                "last": self.indices[-1] if self.indices else None,
            },
            "options": options,
        }

    def _cache_path(
        self,
        cache_dir: str | Path,
        name: str,
        options: dict[str, Any],
    ) -> Path:
        metadata_json = json.dumps(self._metadata(name, options), sort_keys=True)
        digest = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()[:16]
        return Path(cache_dir).absolute() / f"{name}_{digest}.pt"

    @staticmethod
    def _torch_load(path: Path) -> dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_cache(
        self,
        cache_dir: str | Path | None,
        name: str,
        options: dict[str, Any],
        rebuild_cache: bool,
    ) -> dict[str, Any] | None:
        if cache_dir is None:
            return None

        metadata = self._metadata(name, options)
        cache_path = self._cache_path(cache_dir, name, options)
        if not cache_path.exists() or rebuild_cache:
            return None

        payload = self._torch_load(cache_path)
        if payload.get("metadata") != metadata:
            return None
        return payload

    def _save_cache(
        self,
        cache_dir: str | Path | None,
        name: str,
        options: dict[str, Any],
        tensors: dict[str, torch.Tensor | None],
    ) -> None:
        if cache_dir is None:
            return

        cache_path = self._cache_path(cache_dir, name, options)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": self._metadata(name, options),
                **tensors,
            },
            cache_path,
        )


class PushTDataset_MLP(CacheMixin, Dataset):
    def __init__(
        self,
        dataset_dir,
        max_items: int | None = None,
        preload: bool = True,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir).absolute()
        self.dataset = LeRobotDataset(self.dataset_dir)
        self.indices = make_indices(len(self.dataset), max_items)
        self.preload = preload

        self.x = None
        self.actions = None
        if self.preload:
            cache = self._load_or_build_lowdim_cache(
                cache_dir=cache_dir,
                rebuild_cache=rebuild_cache,
                desc="preloading MLP data",
                cache_name="pusht_mlp_lowdim",
                value_key="x",
            )
            self.x = cache.values
            self.actions = cache.actions

    def _load_or_build_lowdim_cache(
        self,
        cache_dir: str | Path | None,
        rebuild_cache: bool,
        desc: str,
        cache_name: str,
        value_key: str,
    ) -> LowdimCache:
        options = {"lowdim_keys": LOWDIM_KEYS, "value_key": value_key}
        payload = self._load_cache(cache_dir, cache_name, options, rebuild_cache)
        if payload is not None:
            return LowdimCache(values=payload[value_key], actions=payload["actions"])

        cache = self._preload_lowdim(desc)
        self._save_cache(
            cache_dir,
            cache_name,
            options,
            {value_key: cache.values, "actions": cache.actions},
        )
        return cache

    def _preload_lowdim(self, desc: str) -> LowdimCache:
        xs = []
        actions = []
        for raw_idx in tqdm(self.indices, desc=desc, unit="sample"):
            sample = self.dataset[raw_idx]
            xs.append(lowdim_from_sample(sample))
            actions.append(action_from_sample(sample))
        return LowdimCache(
            values=torch.stack(xs, dim=0).contiguous(),
            actions=torch.stack(actions, dim=0).contiguous(),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.preload:
            return {"x": self.x[idx], "action": self.actions[idx]}

        sample = self.dataset[self.indices[idx]]
        return {"x": lowdim_from_sample(sample), "action": action_from_sample(sample)}


class PushTDataset(CacheMixin, Dataset):
    def __init__(
        self,
        dataset_dir,
        max_items: int | None = None,
        cache_lowdim: bool = True,
        cache_images: bool = False,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
        image_lru_size: int = 256,
    ):
        self.dataset_dir = Path(dataset_dir).absolute()
        self.dataset = LeRobotDataset(self.dataset_dir)
        self.indices = make_indices(len(self.dataset), max_items)
        self.cache_lowdim = cache_lowdim
        self.cache_images = cache_images
        self.image_lru_size = int(image_lru_size)
        self._image_lru: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

        self.states = None
        self.actions = None
        if self.cache_lowdim:
            lowdim = self._load_or_build_lowdim_cache(
                cache_dir=cache_dir,
                rebuild_cache=rebuild_cache,
                desc="preloading state/action",
            )
            self.states = lowdim.values
            self.actions = lowdim.actions

        self.cam_top = None
        self.cam_side = None
        if self.cache_images:
            images = self._load_or_build_image_cache(
                cache_dir=cache_dir,
                rebuild_cache=rebuild_cache,
                desc="preloading images",
            )
            self.cam_top = images.cam_top
            self.cam_side = images.cam_side

    def _load_or_build_lowdim_cache(
        self,
        cache_dir: str | Path | None,
        rebuild_cache: bool,
        desc: str,
    ) -> LowdimCache:
        cache_name = "pusht_cnn_lowdim"
        options = {"lowdim_keys": LOWDIM_KEYS}
        payload = self._load_cache(cache_dir, cache_name, options, rebuild_cache)
        if payload is not None:
            return LowdimCache(values=payload["states"], actions=payload["actions"])

        cache = self._preload_lowdim(desc)
        self._save_cache(
            cache_dir,
            cache_name,
            options,
            {"states": cache.values, "actions": cache.actions},
        )
        return cache

    def _load_or_build_image_cache(
        self,
        cache_dir: str | Path | None,
        rebuild_cache: bool,
        desc: str,
    ) -> ImageCache:
        cache_name = "pusht_cnn_images"
        options = {"cam_top_key": CAM_TOP_KEY, "cam_side_key": CAM_SIDE_KEY}
        payload = self._load_cache(cache_dir, cache_name, options, rebuild_cache)
        if payload is not None:
            return ImageCache(cam_top=payload["cam_top"], cam_side=payload["cam_side"])

        cache = self._preload_images(desc)
        self._save_cache(
            cache_dir,
            cache_name,
            options,
            {"cam_top": cache.cam_top, "cam_side": cache.cam_side},
        )
        return cache

    def _preload_lowdim(self, desc: str) -> LowdimCache:
        states = []
        actions = []
        for raw_idx in tqdm(self.indices, desc=desc, unit="sample"):
            sample = self.dataset[raw_idx]
            states.append(lowdim_from_sample(sample))
            actions.append(action_from_sample(sample))
        return LowdimCache(
            values=torch.stack(states, dim=0).contiguous(),
            actions=torch.stack(actions, dim=0).contiguous(),
        )

    def _preload_images(self, desc: str) -> ImageCache:
        cam_top = []
        cam_side = []
        for raw_idx in tqdm(self.indices, desc=desc, unit="sample"):
            top, side = self._load_image_pair(raw_idx)
            cam_top.append(top)
            cam_side.append(side)
        return ImageCache(
            cam_top=torch.stack(cam_top, dim=0).contiguous(),
            cam_side=torch.stack(cam_side, dim=0).contiguous(),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = None
        if self.cache_images:
            cam_top = self.cam_top[idx]
            cam_side = self.cam_side[idx]
        else:
            raw_idx = self.indices[idx]
            cam_top, cam_side = self._image_pair(raw_idx)

        if self.cache_lowdim:
            state = self.states[idx]
            action = self.actions[idx]
        else:
            if sample is None:
                sample = self.dataset[self.indices[idx]]
            state = lowdim_from_sample(sample)
            action = action_from_sample(sample)

        return {
            "cam_top": cam_top,
            "cam_side": cam_side,
            "state": state,
            "action": action,
        }

    def _image_pair(self, raw_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.image_lru_size <= 0:
            return self._load_image_pair(raw_idx)

        cached = self._image_lru.get(raw_idx)
        if cached is not None:
            self._image_lru.move_to_end(raw_idx)
            return cached

        pair = self._load_image_pair(raw_idx)
        self._image_lru[raw_idx] = pair
        self._image_lru.move_to_end(raw_idx)
        while len(self._image_lru) > self.image_lru_size:
            self._image_lru.popitem(last=False)
        return pair

    def _load_image_pair(self, raw_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[raw_idx]
        return (
            image_to_chw_float(sample[CAM_TOP_KEY]),
            image_to_chw_float(sample[CAM_SIDE_KEY]),
        )
