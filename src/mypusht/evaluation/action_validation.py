from __future__ import annotations

import numpy as np


def coerce_action(action, action_dim: int = 2, policy_name: str = "Policy") -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < action_dim:
        raise ValueError(
            f"{policy_name} returned shape {action.shape}, expected at least {action_dim} values."
        )
    action = action[:action_dim]
    if not np.all(np.isfinite(action)):
        raise ValueError(f"{policy_name} returned non-finite action: {action}")
    return action
