from dataclasses import dataclass
from pathlib import Path
import numpy as np

from typing import Any

from mypusht.paths import EVAL_OUTPUT_DIR, MODEL_DIR, OUTPUTS_DIR

EVAL_RES_BASE = EVAL_OUTPUT_DIR


def get_next_res_dir():
    base = EVAL_RES_BASE
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("res_")]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    next_num = max(nums) + 1 if nums else 0
    res_dir = base / f"res_{next_num}"
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


def setup_res_dirs(res_dir):
    paths = {
        "results": res_dir / "results",
        "videos": res_dir / "videos"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_res_dirs(res_dir=None):
    if res_dir is None:
        res_dir = get_next_res_dir()
    return setup_res_dirs(res_dir)


@dataclass(frozen=True)
class EvalSplit:
    name: str
    seed_start: int
    episodes: int
    max_steps: int


EVAL_SPLITS = {
    "normal": EvalSplit(
        name="normal",
        seed_start=1000,
        episodes=10,
        max_steps=400,
    ),
    "wide": EvalSplit(
        name="wide",
        seed_start=2000,
        episodes=10,
        max_steps=400,
    )
}


class EvalPolicy:

    def __init__(self, meta):
        self.meta = meta

    def reset_episode(self, instruction: str | None = None) -> None:
        """Clear temporal state before each episode.

        Single-step policies can ignore this.
        ACT / Diffusion adapters should clear cached action chunks here.
        VLA adapters should store or encode the current instruction here.
        """

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        """Return exactly one action for the current env.step()."""
        raise NotImplementedError
    

@dataclass(frozen=True)
class PolicyMeta:
    display_name: str
    family: str
    action_mode: str
    requires_language: bool = False
    ckpt: Path | None = None


POLICY_SPECS = { 
    # 传统单步策略 heuristic, bc_mlp, bc_cnn
    "heuristic": PolicyMeta(
        display_name="Heuristic",
        family="traditional",
        action_mode="single_step",
        requires_language=False,
        ckpt=None,
    ),
    "bc_mlp": PolicyMeta(
        display_name="BC-MLP",
        family="traditional",
        action_mode="single_step",
        requires_language=False,
        ckpt=MODEL_DIR / "bc_mlp_v3.pt",
    ),
    "bc_cnn": PolicyMeta(
        display_name="BC-CNN",
        family="traditional",
        action_mode="single_step",
        requires_language=False,
        ckpt=MODEL_DIR / "bc_cnn_v3.pt",
    ),

    # 序列策略 ACT diffusion_policy
    "act": PolicyMeta(
        display_name="ACT",
        family="sequence_policy",
        action_mode="action_chunk",
        requires_language=False,
        ckpt=MODEL_DIR / "act_v3.pt",
    ),
    "diffusion_policy": PolicyMeta(
        display_name="Diffusion Policy",
        family="sequence_policy",
        action_mode="action_chunk",
        requires_language=False,
        ckpt=MODEL_DIR / "dp_v3.pt",
    ),

    # 语言条件策略 VLA
    "vla_bc": PolicyMeta(
        display_name="VLA-BC",
        family="language_conditioned", 
        action_mode="single_step",
        requires_language=True,
        ckpt=OUTPUTS_DIR / "vla_bc.pt",
    ),
}


INSTRUCTION_SPECS = {
    "default": "push the T block to the target pose",
    "push_left": "push the T block to the left target",
    "push_right": "push the T block to the right target",
    "rotate_clockwise": "rotate the T block clockwise into the target pose",
}
