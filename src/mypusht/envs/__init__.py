from __future__ import annotations

__all__ = ["SimplePushTEnv", "So100PushTEnv", "So100TeleopEnv"]


def __getattr__(name: str):
    if name == "SimplePushTEnv":
        from .simple_pusht_env import SimplePushTEnv

        return SimplePushTEnv
    if name == "So100PushTEnv":
        from .so100_push_t import So100PushTEnv

        return So100PushTEnv
    if name == "So100TeleopEnv":
        from .so100_teleop_env import So100TeleopEnv

        return So100TeleopEnv
    raise AttributeError(name)
