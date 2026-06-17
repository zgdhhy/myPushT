from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short MyPushT environment smoke test.")
    parser.add_argument("--env", choices=["simple", "so100"], default="simple")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="Accepted for CI/headless workflows.")
    parser.add_argument("--xml", type=Path, default=None, help="Optional XML override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env == "simple":
        from mypusht.envs.simple_pusht_env import SimplePushTEnv

        env = SimplePushTEnv(xml_path=args.xml) if args.xml else SimplePushTEnv()
    else:
        from mypusht.envs.so100_push_t import So100PushTEnv

        env = So100PushTEnv(xml_path=args.xml) if args.xml else So100PushTEnv()

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    try:
        for _ in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()

    print(
        f"env={args.env} steps={args.steps} "
        f"success={bool(info.get('success', False))} reward={total_reward:.4f}"
    )


if __name__ == "__main__":
    main()
