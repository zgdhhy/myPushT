import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from mypusht.envs.pose_utils import check_pusht_success, site_xy, yaw_from_body
from mypusht.paths import SIMPLE_XML_PATH


class SimplePushTEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, xml_path=SIMPLE_XML_PATH, max_steps=300, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0

        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1], dtype=np.float32),
            high=np.array([0.1, 0.1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict({
            "image": spaces.Dict({
                "top": spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                "side": spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
            }),
            "state": spaces.Box(-10.0, 10.0, shape=(5,), dtype=np.float32),
            "goal": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
        })

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
        joint_id = self.model.body_jntadr[body_id]
        self.T_joint_pos = self.model.jnt_qposadr[joint_id]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random
        object_x = float(rng.uniform(0.12, 0.22))
        object_y = float(rng.uniform(-0.08, 0.08))
        object_yaw = float(rng.uniform(-0.7, 0.7))

        self.data.qpos[self.T_joint_pos:self.T_joint_pos + 3] = [object_x, object_y, 0.035]
        self.data.qpos[self.T_joint_pos + 3:self.T_joint_pos + 7] = [
            np.cos(object_yaw / 2), 0.0, 0.0, np.sin(object_yaw / 2)
        ]

        self.data.ctrl[:] = [-0.15, 0.0]
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1
        obs = self._get_obs()
        info = self._get_info()
        reward = -info["xy_error"] - 0.2 * info["yaw_error"]
        if info["success"]:
            reward += 10.0

        terminated = info["success"]
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, info

    def _camera(self, name):
        self.renderer.update_scene(self.data, camera=name)
        return self.renderer.render()

    def _get_obs(self):
        object_xy = site_xy(self.model, self.data, "T_block_anchor")
        pusher_xy = site_xy(self.model, self.data, "pusher_site")
        target_xy = site_xy(self.model, self.data, "T_sign_anchor")
        object_yaw = yaw_from_body(self.model, self.data, "T_block")
        target_yaw = yaw_from_body(self.model, self.data, "T_sign")

        state = np.array([
            object_xy[0], object_xy[1], object_yaw,
            pusher_xy[0], pusher_xy[1],
        ], dtype=np.float32)
        goal = np.array([target_xy[0], target_xy[1], target_yaw], dtype=np.float32)

        return {
            "image": {
                "top": self._camera("top_view"),
                "side": self._camera("side_view"),
            },
            "state": state,
            "goal": goal,
        }

    def _get_info(self):
        success, xy_error, yaw_error = check_pusht_success(self.model, self.data)
        return {
            "success": bool(success),
            "xy_error": float(xy_error),
            "yaw_error": float(yaw_error),
        }

    def render(self):
        return self._camera("top_view")

    def close(self):
        self.renderer.close()
