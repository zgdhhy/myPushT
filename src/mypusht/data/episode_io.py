from __future__ import annotations

from pathlib import Path

import numpy as np

from mypusht.paths import RAW_EPISODES_DIR


def next_episode_path(raw_dir: Path = RAW_EPISODES_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_dir.glob("episode_*.npz"))
    if not existing:
        return raw_dir / "episode_0000.npz"

    last_id = int(existing[-1].stem.split("_")[-1])
    return raw_dir / f"episode_{last_id + 1:04d}.npz"


def save_episode(path: Path, buffer: list[dict], fps: int) -> bool:
    if len(buffer) < 2:
        print("skip saving: episode has fewer than 2 frames")
        return False

    arrays = {
        "cam_top": np.stack([x["cam_top"] for x in buffer]).astype(np.uint8),
        "cam_side": np.stack([x["cam_side"] for x in buffer]).astype(np.uint8),
        "state": np.stack([x["state"] for x in buffer]).astype(np.float32),
        "mocap_xy": np.stack([x["mocap_xy"] for x in buffer]).astype(np.float32),
        "object_pose": np.stack([x["object_pose"] for x in buffer]).astype(np.float32),
        "goal_pose": np.stack([x["goal_pose"] for x in buffer]).astype(np.float32),
        "action": np.stack([x["action"] for x in buffer]).astype(np.float32),
        "success": np.array([x["success"] for x in buffer], dtype=np.bool_),
        "xy_error": np.array([x["xy_error"] for x in buffer], dtype=np.float32),
        "yaw_error": np.array([x["yaw_error"] for x in buffer], dtype=np.float32),
        "timestamp": np.array([x["timestamp"] for x in buffer], dtype=np.float32),
    }

    np.savez_compressed(
        path,
        **arrays,
        fps=np.array(fps, dtype=np.int32),
        task=np.array("pushT"),
        action_type=np.array("absolute_mocap_xy"),
    )
    print("saved:", path)
    print("frames:", len(buffer))
    print("final success:", bool(arrays["success"][-1]))
    print("final xy_error:", float(arrays["xy_error"][-1]))
    print("final yaw_error:", float(arrays["yaw_error"][-1]))
    return True
