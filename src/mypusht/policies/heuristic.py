import numpy as np


class HeuristicPolicy:
    def __init__(self, max_step=0.015, approach_offset=0.06):
        self.max_step = float(max_step)
        self.approach_offset = float(approach_offset)

    def predict(self, obs):
        mocap_xy = np.asarray(obs["mocap_xy"], dtype=np.float32)
        object_pose = np.asarray(obs["object_pose"], dtype=np.float32)
        goal_pose = np.asarray(obs["goal_pose"], dtype=np.float32)

        object_xy = object_pose[:2]
        goal_xy = goal_pose[:2]

        push_dir = goal_xy - object_xy
        norm = np.linalg.norm(push_dir)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)

        push_dir = push_dir / norm

        # First move behind the object, then push through it toward the goal.
        behind_object = object_xy - push_dir * self.approach_offset
        push_target = object_xy + push_dir * self.approach_offset

        if np.linalg.norm(mocap_xy - behind_object) > 0.03:
            target_xy = behind_object
        else:
            target_xy = push_target

        delta = target_xy - mocap_xy
        delta = np.clip(delta, -self.max_step, self.max_step)
        return delta.astype(np.float32)