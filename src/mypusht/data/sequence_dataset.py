from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


LOWDIM_KEYS = [
    "observation.state",
    "observation.mocap_xy",
    "observation.object_pose",
    "observation.goal_pose",
]

IMAGE_KEYS = [
    "observation.images.cam_top",
    "observation.images.cam_side",
]

CACHE_VERSION = 1


def to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def image_to_chw_float(value: Any) -> torch.Tensor:
    x = to_tensor(value)

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


def vector_from_keys(sample: dict[str, Any], keys: list[str]) -> torch.Tensor:
    parts = []
    for key in keys:
        parts.append(to_tensor(sample[resolve_sample_key(sample, key)]).reshape(-1))
    return torch.cat(parts, dim=0)


def action_from_sample(sample: dict[str, Any]) -> torch.Tensor:
    if "action" not in sample:
        raise KeyError("missing key: action")
    return to_tensor(sample["action"]).reshape(-1)


def image_stack_from_sample(sample: dict[str, Any], image_keys: list[str]) -> torch.Tensor:
    cams = []
    for key in image_keys:
        if key not in sample:
            raise KeyError(f"missing key: {key}")
        cams.append(image_to_chw_float(sample[key]))
    return torch.stack(cams, dim=0)


def episode_ranges(episode_indices: torch.Tensor) -> list[tuple[int, int]]:
    if episode_indices.numel() == 0:
        return []

    diffs = torch.diff(episode_indices)
    boundaries = torch.where(diffs != 0)[0] + 1
    starts = torch.cat([torch.tensor([0]), boundaries]).tolist()
    ends = torch.cat([boundaries, torch.tensor([len(episode_indices)])]).tolist()
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def build_sequence_indices(
    ranges: list[tuple[int, int]],
    obs_horizon: int,
    action_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_indices = []
    action_indices = []

    for ep_start, ep_end in ranges:
        first_t = ep_start + obs_horizon - 1
        last_t = ep_end - action_horizon + 1
        for t in range(first_t, last_t):
            obs_indices.append(list(range(t - obs_horizon + 1, t + 1)))
            action_indices.append(list(range(t, t + action_horizon)))

    if not obs_indices:
        raise RuntimeError(
            "no valid sequence samples found. Check obs_horizon, action_horizon, "
            "dataset length, and episode_index fields."
        )

    return (
        torch.tensor(obs_indices, dtype=torch.long),
        torch.tensor(action_indices, dtype=torch.long),
    )


@dataclass(frozen=True)
class FrameCache:
    states: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    images: torch.Tensor | None = None


class PushTSequenceDataset(Dataset):
    """ACT-style sequence dataset for PushT.

    Returned sample shapes:
        states:  (obs_horizon, state_dim)
        images:  (obs_horizon, num_cameras, channels, height, width)
        actions: (action_horizon, action_dim)
    """

    def __init__(
        self,
        dataset_dir,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        max_items: int | None = None,
        image_keys: list[str] | None = None,
        lowdim_keys: list[str] | None = None,
        cache_lowdim: bool = True,
        cache_images: bool = False,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
        image_lru_size: int = 256,
    ):
        if obs_horizon <= 0:
            raise ValueError(f"obs_horizon must be positive, got {obs_horizon}")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")

        self.dataset_dir = Path(dataset_dir).absolute()
        self.dataset = LeRobotDataset(self.dataset_dir)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.image_keys = list(image_keys or IMAGE_KEYS)
        self.lowdim_keys = list(lowdim_keys or LOWDIM_KEYS)
        self.cache_lowdim = bool(cache_lowdim)
        self.cache_images = bool(cache_images)
        self.image_lru_size = int(image_lru_size)
        self._image_lru: OrderedDict[int, torch.Tensor] = OrderedDict()

        self.total_frames = len(self.dataset)
        if max_items is not None:
            self.total_frames = min(self.total_frames, int(max_items))
        if self.total_frames <= 0:
            raise RuntimeError("dataset is empty after applying max_items")

        ep_indices = self.dataset.hf_dataset["episode_index"]
        ep_indices = torch.as_tensor(ep_indices, dtype=torch.long)[: self.total_frames]
        ranges = episode_ranges(ep_indices)

        self.obs_indices, self.action_indices = build_sequence_indices(
            ranges,
            self.obs_horizon,
            self.action_horizon,
        )
        self.indices = self.action_indices[:, 0].tolist()

        self.cache = self._load_or_build_frame_cache(cache_dir, rebuild_cache)

    def _metadata(self) -> dict[str, Any]:
        return {
            "version": CACHE_VERSION,
            "dataset_dir": str(self.dataset_dir),
            "total_frames": self.total_frames,
            "lowdim_keys": self.lowdim_keys,
            "image_keys": self.image_keys,
            "cache_lowdim": self.cache_lowdim,
            "cache_images": self.cache_images,
        }

    def _cache_path(self, cache_dir: str | Path) -> Path:
        metadata_json = json.dumps(self._metadata(), sort_keys=True)
        digest = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()[:16]
        return Path(cache_dir).absolute() / f"pusht_sequence_cache_{digest}.pt"

    def _load_or_build_frame_cache(
        self,
        cache_dir: str | Path | None,
        rebuild_cache: bool,
    ) -> FrameCache:
        metadata = self._metadata()
        cache_path = self._cache_path(cache_dir) if cache_dir is not None else None

        if cache_path is not None and cache_path.exists() and not rebuild_cache:
            payload = self._torch_load(cache_path)
            if payload.get("metadata") == metadata:
                return FrameCache(
                    states=payload.get("states"),
                    actions=payload.get("actions"),
                    images=payload.get("images"),
                )

        frame_cache = self._build_frame_cache()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": metadata,
                    "states": frame_cache.states,
                    "actions": frame_cache.actions,
                    "images": frame_cache.images,
                },
                cache_path,
            )

        return frame_cache

    @staticmethod
    def _torch_load(path: Path) -> dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _build_frame_cache(self) -> FrameCache:
        if not self.cache_lowdim and not self.cache_images:
            return FrameCache()

        states = []
        actions = []
        images = []

        for raw_idx in tqdm(
            range(self.total_frames),
            desc="building PushT sequence cache",
            unit="frame",
            disable=self.total_frames < 64,
        ):
            sample = self.dataset[raw_idx]
            if self.cache_lowdim:
                states.append(vector_from_keys(sample, self.lowdim_keys))
                actions.append(action_from_sample(sample))
            if self.cache_images:
                images.append(image_stack_from_sample(sample, self.image_keys))

        return FrameCache(
            states=torch.stack(states, dim=0).contiguous() if states else None,
            actions=torch.stack(actions, dim=0).contiguous() if actions else None,
            images=torch.stack(images, dim=0).contiguous() if images else None,
        )

    def __len__(self) -> int:
        return len(self.obs_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs_idx = self.obs_indices[idx]
        action_idx = self.action_indices[idx]

        return {
            "states": self._states(obs_idx),
            "images": self._images(obs_idx),
            "actions": self._actions(action_idx),
        }

    def _states(self, indices: torch.Tensor) -> torch.Tensor:
        if self.cache.states is not None:
            return self.cache.states.index_select(0, indices)

        return torch.stack(
            [
                vector_from_keys(self.dataset[int(raw_idx)], self.lowdim_keys)
                for raw_idx in indices.tolist()
            ],
            dim=0,
        )

    def _actions(self, indices: torch.Tensor) -> torch.Tensor:
        if self.cache.actions is not None:
            return self.cache.actions.index_select(0, indices)

        return torch.stack(
            [
                action_from_sample(self.dataset[int(raw_idx)])
                for raw_idx in indices.tolist()
            ],
            dim=0,
        )

    def _images(self, indices: torch.Tensor) -> torch.Tensor:
        if self.cache.images is not None:
            return self.cache.images.index_select(0, indices)

        return torch.stack(
            [self._image_frame(int(raw_idx)) for raw_idx in indices.tolist()],
            dim=0,
        )

    def _image_frame(self, raw_idx: int) -> torch.Tensor:
        if self.image_lru_size <= 0:
            return image_stack_from_sample(self.dataset[raw_idx], self.image_keys)

        cached = self._image_lru.get(raw_idx)
        if cached is not None:
            self._image_lru.move_to_end(raw_idx)
            return cached

        image = image_stack_from_sample(self.dataset[raw_idx], self.image_keys)
        self._image_lru[raw_idx] = image
        self._image_lru.move_to_end(raw_idx)
        while len(self._image_lru) > self.image_lru_size:
            self._image_lru.popitem(last=False)
        return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-cache-lowdim", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--image-lru-size", type=int, default=256)
    args = parser.parse_args()

    ds = PushTSequenceDataset(
        args.dataset,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        max_items=args.max_items,
        cache_lowdim=not args.no_cache_lowdim,
        cache_images=args.cache_images,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        image_lru_size=args.image_lru_size,
    )

    sample = ds[0]
    print("num sequence samples:", len(ds))
    print("states:", tuple(sample["states"].shape))
    print("images:", tuple(sample["images"].shape))
    print("actions:", tuple(sample["actions"].shape))
    print("first action:", sample["actions"][0].numpy())


if __name__ == "__main__":
    main()
