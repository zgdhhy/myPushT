from __future__ import annotations

import cv2
import numpy as np

from mypusht.envs.so100_teleop_env import StepInfo
from mypusht.teleop.config import OBS_SHAPE


WINDOW_NAME = "phase3 preview"
_WINDOW_CREATED = False
_CACHED_SCALE: int | None = None
_TOP_BGR: np.ndarray | None = None
_SIDE_BGR: np.ndarray | None = None
_TOP_RESIZED: np.ndarray | None = None
_SIDE_RESIZED: np.ndarray | None = None


def show_preview(
    cam_top: np.ndarray,
    cam_side: np.ndarray,
    info: StepInfo,
    mocap_xy: np.ndarray,
    recording: bool,
    frame_count: int,
    preview_scale: int
) -> None:
    global _WINDOW_CREATED, _CACHED_SCALE
    global _TOP_BGR, _SIDE_BGR, _TOP_RESIZED, _SIDE_RESIZED

    if _CACHED_SCALE != preview_scale:
        _CACHED_SCALE = preview_scale
        cvt_h, cvt_w = OBS_SHAPE[0], OBS_SHAPE[1]
        new_w = OBS_SHAPE[1] * preview_scale
        new_h = OBS_SHAPE[0] * preview_scale
        _TOP_BGR = np.empty((cvt_h, cvt_w, 3), dtype=np.uint8)
        _SIDE_BGR = np.empty((cvt_h, cvt_w, 3), dtype=np.uint8)
        _TOP_RESIZED = np.empty((new_h, new_w, 3), dtype=np.uint8)
        _SIDE_RESIZED = np.empty((new_h, new_w, 3), dtype=np.uint8)

    cv2.cvtColor(cam_top, cv2.COLOR_RGB2BGR, dst=_TOP_BGR)
    cv2.cvtColor(cam_side, cv2.COLOR_RGB2BGR, dst=_SIDE_BGR)
    size = (OBS_SHAPE[1] * preview_scale, OBS_SHAPE[0] * preview_scale)
    cv2.resize(_TOP_BGR, size, dst=_TOP_RESIZED)
    cv2.resize(_SIDE_BGR, size, dst=_SIDE_RESIZED)

    color = (0, 0, 255) if recording else (0, 180, 0)
    cv2.putText(
        _TOP_RESIZED,
        f"REC={recording} frames={frame_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        _TOP_RESIZED,
        f"success={info.success} xy={info.xy_error:.4f} yaw={info.yaw_error:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        _TOP_RESIZED,
        f"mocap_xy=({mocap_xy[0]:.3f}, {mocap_xy[1]:.3f})",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    combined = np.hstack([_TOP_RESIZED, _SIDE_RESIZED])
    if not _WINDOW_CREATED:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        _WINDOW_CREATED = True
    cv2.imshow(WINDOW_NAME, combined)
    cv2.waitKey(1)
