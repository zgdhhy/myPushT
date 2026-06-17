from __future__ import annotations


from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mypusht.evaluation.action_validation import coerce_action
from mypusht.evaluation.config import POLICY_SPECS, EvalPolicy, PolicyMeta

from mypusht.envs.so100_push_t import So100PushTEnv

from mypusht.policies.heuristic import HeuristicPolicy
from mypusht.policies.bc_mlp import BCMLPPolicy
from mypusht.policies.bc_cnn import BCCNNPolicy

from mypusht.policies.act import ACTPolicy
from mypusht.policies.diffusion_policy import DPPolicy


def load_env(split: str, max_steps: int) -> gym.Env:
    raw_env = So100PushTEnv(max_steps=max_steps)
    if split == "wide":
        raw_env.pos_random_range = 0.07
    else:
        raw_env.pos_random_range = 0.05
    return raw_env


class SingleStepPolicyAdapter(EvalPolicy):
    def __init__(self, policy, meta: PolicyMeta, action_dim: int = 2):
        super().__init__(meta)
        self.policy = policy
        self.action_dim = int(action_dim)

    def predict(self, obs: dict) -> np.ndarray:
        action = self.policy.predict(obs)
        return coerce_action(action, self.action_dim, policy_name="Policy")


class SequencePolicyAdapter(EvalPolicy):
    def __init__(self, policy, meta: PolicyMeta, action_dim: int = 2):
        super().__init__(meta)
        self.policy = policy
        self.action_dim = int(action_dim)

    def reset_episode(self, instruction: str | None = None) -> None:
        if hasattr(self.policy, "reset_episode"):
            self.policy.reset_episode(instruction=instruction)
        elif hasattr(self.policy, "reset"):
            self.policy.reset()


    def predict(self, obs: dict) -> np.ndarray:
        action = self.policy.predict(obs)
        return coerce_action(action, self.action_dim, policy_name="Policy")


class LanguageConditionedPolicyAdapter(EvalPolicy):
    """Adapter for Phase 6 VLA-style policies."""
    def __init__(self, policy, meta: PolicyMeta, action_dim: int = 2):
        super().__init__(meta)
        self.policy = policy
        self.action_dim = int(action_dim)
        self.instruction: str | None = None

    def reset_episode(self, instruction: str | None = None) -> None:
        self.instruction = instruction
        if hasattr(self.policy, "reset_episode"):
            self.policy.reset_episode(instruction=instruction)

    def predict(self, obs: dict) -> np.ndarray:
        obs = dict(obs)
        obs["language_instruction"] = self.instruction

        # TODO: 按你的 VLA / language-conditioned policy 实现改这里。
        # 可能的接口示例：
        #   action = self.policy.predict(obs, instruction=self.instruction)
        #   action = self.policy.predict(obs)
        try:
            action = self.policy.predict(obs, instruction=self.instruction)
        except TypeError:
            action = self.policy.predict(obs)

        return coerce_action(action, self.action_dim, policy_name="VLA policy")


def load_policy(policy_name: str, ckpt: str | Path | None = None, device: torch.device = None, **kwargs) -> EvalPolicy:
    meta: PolicyMeta = POLICY_SPECS[policy_name]
    ckpt = ckpt if ckpt is not None else meta.ckpt
    
    if meta.family == "traditional":
        if policy_name == "heuristic":
            raw_policy = HeuristicPolicy()
        elif policy_name == "bc_mlp":
            raw_policy = BCMLPPolicy(ckpt, device=device)
        elif policy_name == "bc_cnn":
            raw_policy = BCCNNPolicy(ckpt, device=device)
        else:
            raw_policy = HeuristicPolicy()
        return SingleStepPolicyAdapter(raw_policy, meta=meta)
    
    if meta.family == "sequence_policy":
        exec_horizon = kwargs.get("exec_horizon", 8)
        if policy_name == "act":
            raw_policy = ACTPolicy(ckpt, device=device, exec_horizon=exec_horizon)
        elif policy_name == "diffusion_policy":
            raw_policy = DPPolicy(ckpt, device=device, exec_horizon=exec_horizon)
        return SequencePolicyAdapter(raw_policy, meta=meta)

    if meta.family == "language_conditioned":
        # TODO: 改成 language_conditioned 的真实加载方式。
        # Example:
        # raw_policy = VLAPolicy.load_from_checkpoint(ckpt, device=device)
        # return LanguageConditionedPolicyAdapter(raw_policy, meta=meta, action_dim=2)
        raise NotImplementedError("Language-conditioned policy loading is roadmap-only in V1.")

    raise NotImplementedError(
        f"Unknown policy family for {policy_name}: {meta.family}"
    )
