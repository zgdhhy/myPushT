import math

import numpy as np

from mypusht.envs.pose_utils import is_pose_success, pusht_pose_error, wrap_to_pi


def test_wrap_to_pi_stays_in_closed_range():
    values = [wrap_to_pi(x) for x in [-4 * math.pi, -math.pi, 0.0, math.pi, 4 * math.pi]]
    assert all(-math.pi <= x <= math.pi for x in values)
    assert wrap_to_pi(3 * math.pi) == -math.pi


def test_pusht_pose_error_and_success():
    object_pose = np.array([0.25, 0.01, 0.05], dtype=np.float32)
    goal_pose = np.array([0.26, 0.02, 0.10], dtype=np.float32)

    success, xy_error, yaw_error = is_pose_success(object_pose, goal_pose)

    assert success
    assert xy_error < 0.025
    assert yaw_error < 0.20
    assert pusht_pose_error(object_pose, goal_pose) == (xy_error, yaw_error)


def test_pusht_pose_failure_on_yaw():
    success, _, yaw_error = is_pose_success(
        np.array([0.0, 0.0, 0.50], dtype=np.float32),
        np.array([0.0, 0.0, 0.00], dtype=np.float32),
    )
    assert not success
    assert yaw_error > 0.20
