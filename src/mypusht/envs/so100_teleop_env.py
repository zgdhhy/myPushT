from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from mypusht.envs.pose_utils import check_pusht_success, site_xy, yaw_from_body
from mypusht.paths import SO100_XML_PATH


@dataclass(frozen=True)
class StepInfo:
    success: bool
    xy_error: float
    yaw_error: float


OBS_SHAPE = (224, 224, 3)
ACTUATOR_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
XML_PATH = SO100_XML_PATH


class So100TeleopEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        xml_path=XML_PATH,
        obs_shape=OBS_SHAPE,
        move_speed: float = 0.1,
        workspace_x: tuple[float, float] = (0.05, 0.45),
        workspace_y: tuple[float, float] = (-0.25, 0.25),
        max_steps: int | None = None,
        render_mode: str | None = None,
        image_observation: bool = True,
    ) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=obs_shape[0], width=obs_shape[1])

        self.obs_shape = obs_shape
        self.move_speed = float(move_speed)
        self.workspace_x = workspace_x
        self.workspace_y = workspace_y
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.image_observation = image_observation
        self.current_step = 0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        observation_spaces = {
            "robot_state": spaces.Box(-10.0, 10.0, shape=(len(ACTUATOR_NAMES),), dtype=np.float32),
            "mocap_xy": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
            "object_pose": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
            "goal_pose": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
        }
        if image_observation:
            observation_spaces["image"] = spaces.Dict({
                "top": spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8),
                "side": spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8),
            })
        self.observation_space = spaces.Dict(observation_spaces)

        self.actuator_ids = self._resolve_actuator_ids()
        self.actuator_qpos_addrs = self._resolve_actuator_qpos_addrs()
        self.mocap_id = self.model.body("target_mocap").mocapid[0]
        self.t_qpos_addr = self._body_freejoint_qpos_addr("T_block")

    def _resolve_actuator_ids(self) -> np.ndarray:
        actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]
        if any(x < 0 for x in actuator_ids):
            raise RuntimeError(f"missing actuator in {ACTUATOR_NAMES}: {actuator_ids}")
        return np.array(actuator_ids, dtype=np.int32)

    def _resolve_actuator_qpos_addrs(self) -> np.ndarray:
        qpos_addrs = []
        for actuator_id in self.actuator_ids:
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            if joint_id < 0:
                raise RuntimeError(f"actuator {actuator_id} is not bound to a joint")
            qpos_addrs.append(self.model.jnt_qposadr[joint_id])
        return np.array(qpos_addrs, dtype=np.int32)

    def _body_freejoint_qpos_addr(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"missing body: {body_name}")
        joint_id = self.model.body_jntadr[body_id]
        if joint_id < 0:
            raise RuntimeError(f"body {body_name} has no joint")
        return int(self.model.jnt_qposadr[joint_id])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        options = options or {}
        pos_random_range = float(options.get("pos_random_range", 0.08))

        key_id = 0
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.data.mocap_pos[:] = self.model.key_mpos[key_id]
        self.data.mocap_quat[:] = self.model.key_mquat[key_id]

        self.data.qpos[self.t_qpos_addr:self.t_qpos_addr + 2] = [
            self.np_random.uniform(0.25 - pos_random_range, 0.25 + pos_random_range),
            self.np_random.uniform(-pos_random_range, pos_random_range),
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
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._apply_mocap_action(action)
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.actuator_qpos_addrs]

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = -info["xy_error"] - 0.2 * info["yaw_error"]
        if info["success"]:
            reward += 10.0

        terminated = bool(info["success"])
        truncated = self.max_steps is not None and self.current_step >= self.max_steps
        return obs, float(reward), terminated, truncated, info

    def _apply_mocap_action(self, action: np.ndarray) -> None:
        delta = np.array([action[0], action[1], 0.0], dtype=np.float32)
        self.data.mocap_pos[self.mocap_id] += delta * self.move_speed * self.model.opt.timestep
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
            "robot_state": self.robot_state(),
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
        if self.image_observation:
            obs["image"] = {
                "top": self.render_camera("top_view"),
                "side": self.render_camera("side_view"),
            }
        return obs

    def _get_info(self) -> dict:
        success, xy_error, yaw_error = check_pusht_success(self.model, self.data)
        return {
            "success": bool(success),
            "xy_error": float(xy_error),
            "yaw_error": float(yaw_error),
        }

    def render_camera(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def render(self):
        return self.render_camera("top_view")

    def mocap_xy(self) -> np.ndarray:
        return self.data.mocap_pos[self.mocap_id][:2].copy().astype(np.float32)

    def robot_state(self) -> np.ndarray:
        return self.data.qpos[self.actuator_qpos_addrs].copy().astype(np.float32)

    @staticmethod
    def step_info(info: dict) -> StepInfo:
        return StepInfo(
            success=bool(info["success"]),
            xy_error=float(info["xy_error"]),
            yaw_error=float(info["yaw_error"]),
        )

    def close(self) -> None:
        self.renderer.close()
