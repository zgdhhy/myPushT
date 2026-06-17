import argparse
from pathlib import Path

import numpy as np

from mypusht.paths import LEROBOT_DATASET_DIR, RAW_EPISODES_DIR

DEFAULT_RAW_DIR = RAW_EPISODES_DIR
DEFAULT_OUT_DIR = LEROBOT_DATASET_DIR
OBS_IMAGES_SHAPE = (224, 224, 3)
ACTUATOR_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")


def validate_episode(path, arrays):
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path.name}: inconsistent frame counts: {lengths}")
    if next(iter(lengths.values())) < 2:
        raise ValueError(f"{path.name}: at least 2 frames are required")

    image_shape_text = (
        f"(T,{OBS_IMAGES_SHAPE[0]},{OBS_IMAGES_SHAPE[1]},{OBS_IMAGES_SHAPE[2]})"
    )
    shape_checks = [
        ("cam_top", arrays["cam_top"].shape[1:] == OBS_IMAGES_SHAPE, image_shape_text),
        ("cam_side", arrays["cam_side"].shape[1:] == OBS_IMAGES_SHAPE, image_shape_text),
        ("state", arrays["state"].shape[1:] == (5,), "(T,5)"),
        ("mocap_xy", arrays["mocap_xy"].shape[1:] == (2,), "(T,2)"),
        ("object_pose", arrays["object_pose"].shape[1:] == (3,), "(T,3)"),
        ("goal_pose", arrays["goal_pose"].shape[1:] == (3,), "(T,3)"),
    ]
    for key, ok, expected in shape_checks:
        if not ok:
            raise ValueError(
                f"{path.name}: {key} expected {expected}, got {arrays[key].shape}"
            )


def delta_actions_from_mocap(mocap_xy: np.ndarray) -> np.ndarray:
    """Return the public MyPushT action contract: raw delta mocap XY."""
    mocap_xy = np.asarray(mocap_xy, dtype=np.float32)
    if mocap_xy.ndim != 2 or mocap_xy.shape[1] != 2:
        raise ValueError(f"mocap_xy must have shape (T, 2), got {mocap_xy.shape}")
    if mocap_xy.shape[0] < 2:
        raise ValueError("mocap_xy needs at least two frames to build actions")
    return np.diff(mocap_xy, axis=0).astype(np.float32)


def create_dataset(repo_path, fps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.cam_top": {
            "dtype": "video",
            "shape": OBS_IMAGES_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_side": {
            "dtype": "video",
            "shape": OBS_IMAGES_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (5,),
            "names": list(ACTUATOR_NAMES),
        },
        "observation.mocap_xy": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["mocap_x", "mocap_y"],
        },
        "observation.object_pose": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["object_x", "object_y", "object_yaw"],
        },
        "observation.goal_pose": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["goal_x", "goal_y", "goal_yaw"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["delta_mocap_x", "delta_mocap_y"],
        },
    }

    return LeRobotDataset.create(
        repo_id=str(repo_path),
        fps=fps,
        robot_type="so100_arm",
        features=features,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    paths = sorted(args.raw_dir.glob("episode_*.npz"))
    if not paths:
        raise SystemExit(f"No raw episodes found in {args.raw_dir}")

    if args.out_dir.exists():
        raise SystemExit(
            f"Output dataset already exists: {args.out_dir}\n"
            "Move it away or choose a new --out_dir to avoid mixing experiments."
        )

    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset(args.out_dir.absolute(), fps=args.fps)

    for episode_index, path in enumerate(paths):
        with np.load(path, allow_pickle=True) as data:
            cam_top = data["cam_top"]
            cam_side = data["cam_side"]
            state = data["state"].astype(np.float32)
            mocap_xy = data["mocap_xy"].astype(np.float32)
            object_pose = data["object_pose"].astype(np.float32)
            goal_pose = data["goal_pose"].astype(np.float32)

            arrays = {
                "cam_top": cam_top,
                "cam_side": cam_side,
                "state": state,
                "mocap_xy": mocap_xy,
                "object_pose": object_pose,
                "goal_pose": goal_pose,
            }
            validate_episode(path, arrays)
            action = delta_actions_from_mocap(mocap_xy)
            frame_count = len(action)

            print(
                f"converting {path.name}: raw_frames={len(mocap_xy)} "
                f"converted_frames={frame_count}"
            )
            for i in range(frame_count):
                dataset.add_frame({
                    "observation.images.cam_top": cam_top[i],
                    "observation.images.cam_side": cam_side[i],
                    "observation.state": state[i],
                    "observation.mocap_xy": mocap_xy[i],
                    "observation.object_pose": object_pose[i],
                    "observation.goal_pose": goal_pose[i],
                    "action": action[i],
                    "task": "pushT",
                })

            dataset.save_episode()
            print(f"saved LeRobot episode {episode_index}")

    print("done")
    print("dataset:", args.out_dir.absolute())
    print("episodes:", len(paths))


if __name__ == "__main__":
    main()
