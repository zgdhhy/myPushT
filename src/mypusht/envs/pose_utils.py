from __future__ import annotations

import numpy as np


def _require_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mujoco is required for MyPushT MuJoCo environment helpers. "
            "Install the project environment with `conda env create -f environment.yml`."
        ) from exc
    return mujoco


# 根据actuator名称解析出actuator id，并检查是否存在
def resolve_actuator_ids(model, actuator_names) -> np.ndarray:
    mujoco = _require_mujoco()
    actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in actuator_names
    ]
    if any(x < 0 for x in actuator_ids):
        raise RuntimeError(f"missing actuator in {actuator_names}: {actuator_ids}")
    return np.array(actuator_ids, dtype=np.int32)

# 根据actuator id解析出对应的qpos地址，并检查actuator是否绑定了joint
def resolve_actuator_qpos_addrs(model, actuator_ids) -> np.ndarray:
    qpos_addrs = []
    for actuator_id in actuator_ids:
        joint_id = model.actuator_trnid[actuator_id, 0]
        if joint_id < 0:
            raise RuntimeError(f"actuator {actuator_id} is not bound to a joint")
        qpos_addrs.append(model.jnt_qposadr[joint_id])
    return np.array(qpos_addrs, dtype=np.int32)

# 根据body名称解析出body id，并检查是否存在；再根据body id解析出对应的qpos地址，并检查body是否绑定了joint
def body_freejoint_qpos_addr(model, body_name: str) -> int:
    mujoco = _require_mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"missing body: {body_name}")
    joint_id = model.body_jntadr[body_id]
    if joint_id < 0:
        raise RuntimeError(f"body {body_name} has no joint")
    return int(model.jnt_qposadr[joint_id])

# 根据body名称解析出body id，并检查是否存在；再根据body id解析出对应的yaw角
def yaw_from_body(model, data, body_name):
    mujoco = _require_mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mat = data.xmat[body_id].reshape(3, 3)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))

# 将角度规范化到[-pi, pi]范围内
def wrap_to_pi(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def pusht_pose_error(
    object_pose: np.ndarray,
    goal_pose: np.ndarray,
) -> tuple[float, float]:
    object_pose = np.asarray(object_pose, dtype=np.float32).reshape(3)
    goal_pose = np.asarray(goal_pose, dtype=np.float32).reshape(3)
    xy_error = float(np.linalg.norm(object_pose[:2] - goal_pose[:2]))
    yaw_error = abs(wrap_to_pi(float(object_pose[2] - goal_pose[2])))
    return xy_error, yaw_error


def is_pose_success(
    object_pose: np.ndarray,
    goal_pose: np.ndarray,
    pos_tol: float = 0.025,
    yaw_tol: float = 0.20,
) -> tuple[bool, float, float]:
    xy_error, yaw_error = pusht_pose_error(object_pose, goal_pose)
    return xy_error < pos_tol and yaw_error < yaw_tol, xy_error, yaw_error

# 根据site名称解析出site id，并检查是否存在；再根据site id解析出对应的xy位置
def site_xy(model, data, site_name):
    mujoco = _require_mujoco()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"missing site: {site_name}")
    return data.site_xpos[site_id][:2].copy()

# 检查push任务是否成功，成功的条件是物体和目标之间的xy位置误差小于pos_tol，并且yaw误差小于yaw_tol
def check_pusht_success(model, data, pos_tol=0.025, yaw_tol=0.20):
    object_xy = site_xy(model, data, "T_block_anchor")
    target_xy = site_xy(model, data, "T_sign_anchor")
    object_yaw = yaw_from_body(model, data, "T_block")
    target_yaw = yaw_from_body(model, data, "T_sign")
    return is_pose_success(
        np.array([object_xy[0], object_xy[1], object_yaw], dtype=np.float32),
        np.array([target_xy[0], target_xy[1], target_yaw], dtype=np.float32),
        pos_tol=pos_tol,
        yaw_tol=yaw_tol,
    )
