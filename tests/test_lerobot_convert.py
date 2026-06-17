import numpy as np
import pytest

from mypusht.data.lerobot_convert import delta_actions_from_mocap, validate_episode


def test_delta_actions_from_mocap_uses_raw_xy_difference():
    mocap_xy = np.array([[0.1, 0.2], [0.15, 0.18], [0.16, 0.20]], dtype=np.float32)

    action = delta_actions_from_mocap(mocap_xy)

    np.testing.assert_allclose(action, [[0.05, -0.02], [0.01, 0.02]], atol=1e-6)
    assert action.dtype == np.float32


def test_delta_actions_rejects_absolute_or_flat_shape():
    with pytest.raises(ValueError, match="shape"):
        delta_actions_from_mocap(np.array([0.1, 0.2], dtype=np.float32))


def test_validate_episode_rejects_mismatched_lengths(tmp_path):
    arrays = {
        "cam_top": np.zeros((3, 224, 224, 3), dtype=np.uint8),
        "cam_side": np.zeros((2, 224, 224, 3), dtype=np.uint8),
        "state": np.zeros((3, 5), dtype=np.float32),
        "mocap_xy": np.zeros((3, 2), dtype=np.float32),
        "object_pose": np.zeros((3, 3), dtype=np.float32),
        "goal_pose": np.zeros((3, 3), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="inconsistent"):
        validate_episode(tmp_path / "episode_0000.npz", arrays)
