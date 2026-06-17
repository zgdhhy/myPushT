from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from mypusht.envs.pose_utils import (
    body_freejoint_qpos_addr,
    check_pusht_success,
    resolve_actuator_ids,
    resolve_actuator_qpos_addrs,
    site_xy,
    yaw_from_body,
)
from mypusht.paths import SO100_XML_PATH

OBS_SHAPE = (224, 224, 3)
ACTUATOR_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
XML_PATH = SO100_XML_PATH


class So100PushTEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        xml_path=XML_PATH,
        obs_shape=OBS_SHAPE,
        workspace_x: tuple[float, float] = (0.05, 0.50),
        workspace_y: tuple[float, float] = (-0.25, 0.25),
        max_action: float = 0.05,
        max_steps: int | None = None,
    ) -> None:
        
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=obs_shape[0], width=obs_shape[1])

        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.max_steps = max_steps

        self.current_step = 0
        self.pos_random_range = 0.05

        observation_spaces = {
            "images": spaces.Dict({
                "cam_top": spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8),
                "cam_side": spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8),
            }),
            "robot_state": spaces.Box(-10.0, 10.0, shape=(len(ACTUATOR_NAMES),), dtype=np.float32),
            "mocap_xy": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
            "object_pose": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
            "goal_pose": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
        }
        self.observation_space = spaces.Dict(observation_spaces)

        self.action_space = spaces.Box(
            low=np.array([-max_action, -max_action], dtype=np.float32),
            high=np.array([max_action, max_action], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # 获取actuator的id 用来访问data.ctrl
        self.actuator_ids = resolve_actuator_ids(self.model, ACTUATOR_NAMES)
        # 获取actuator的qpos地址 用来访问data.qpos
        self.actuator_qpos_addrs = resolve_actuator_qpos_addrs(self.model, self.actuator_ids)
        # 获取mocap的id 用来访问data.mocap_pos和data.mocap_quat
        self.mocap_id = self.model.body("target_mocap").mocapid[0]
        # 获取T_block的qpos地址 用来在reset时随机初始化位置
        self.t_qpos_addr = body_freejoint_qpos_addr(self.model, "T_block")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        options = options or {}
        _pos_random_range = float(options.get("pos_random_range", self.pos_random_range))

        # 初始化keyframe来设置mocap的初始位置和姿态
        key_id = 0
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.data.mocap_pos[:] = self.model.key_mpos[key_id]
        self.data.mocap_quat[:] = self.model.key_mquat[key_id]

        # 随机初始化T_block的位置和姿态
        self.data.qpos[self.t_qpos_addr:self.t_qpos_addr + 2] = [
            self.np_random.uniform(0.25 - _pos_random_range, 0.25 + _pos_random_range),
            self.np_random.uniform(-_pos_random_range, _pos_random_range),
        ]
        self.data.qpos[self.t_qpos_addr + 2] = 0.005
        yaw = self.np_random.uniform(-0.5, 0.5)
        self.data.qpos[self.t_qpos_addr + 3:self.t_qpos_addr + 7] = [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action):
        # 将动作转换为numpy数组并裁剪到动作空间的范围内
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 将动作应用到mocap上
        self._apply_mocap_action(action)
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.actuator_qpos_addrs]

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        # 计算观察值、奖励、终止条件和信息
        obs = self._get_obs()
        info = self._get_info()
        reward = -info["xy_error"] - 0.2 * info["yaw_error"]
        if info["success"]:
            reward += 10.0

        terminated = info["success"]
        truncated = self.max_steps is not None and self.current_step >= self.max_steps
        return obs, float(reward), terminated, truncated, info

    def _apply_mocap_action(self, action: np.ndarray) -> None:
        delta = np.array([action[0], action[1], 0.0], dtype=np.float32)
        # 根据动作更新mocap的位置
        self.data.mocap_pos[self.mocap_id] += delta
        
        # 将mocap的位置限制在工作空间内
        self.data.mocap_pos[self.mocap_id, 0] = np.clip(
            self.data.mocap_pos[self.mocap_id, 0],
            self.workspace_x[0],
            self.workspace_x[1],
        )
        self.data.mocap_pos[self.mocap_id, 1] = np.clip(
            self.data.mocap_pos[self.mocap_id, 1],
            self.workspace_y[0],
            self.workspace_y[1],
        )

    def _get_obs(self) -> dict:
        object_xy = site_xy(self.model, self.data, "T_block_anchor")
        target_xy = site_xy(self.model, self.data, "T_sign_anchor")
        obs = {
            "images": {
                "cam_top": self.render_camera("top_view"),
                "cam_side": self.render_camera("side_view"),
            },
            "state": self.robot_state(),
            "mocap_xy": self.mocap_xy(),
            "object_pose": np.array([
                object_xy[0],
                object_xy[1],
                yaw_from_body(self.model, self.data, "T_block"),
            ], dtype=np.float32),
            "goal_pose": np.array([
                target_xy[0],
                target_xy[1],
                yaw_from_body(self.model, self.data, "T_sign"),
            ], dtype=np.float32),
        }
        return obs

    def _get_info(self) -> dict:
        success, xy_error, yaw_error = check_pusht_success(self.model, self.data)
        return {
            "success": bool(success),
            "xy_error": float(xy_error),
            "yaw_error": float(yaw_error),
        }

    def mocap_xy(self) -> np.ndarray:
        return self.data.mocap_pos[self.mocap_id][:2].copy().astype(np.float32)

    def robot_state(self) -> np.ndarray:
        return self.data.qpos[self.actuator_qpos_addrs].copy().astype(np.float32)

    def render_camera(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def render(self):
        return self.render_camera("top_view")
    
    def close(self) -> None:
        self.renderer.close()
