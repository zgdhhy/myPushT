from __future__ import annotations

import argparse
import sys


TRAIN_MODULES = {
    "bc-mlp": "mypusht.training.train_bc_mlp",
    "bc-cnn": "mypusht.training.train_bc_cnn",
    "act": "mypusht.training.train_act",
    "diffusion-policy": "mypusht.training.train_dp",
}

POLICY_USAGE = {
    "bc-mlp": "mypusht-train bc-mlp --dataset outputs/phase3/lerobot_dataset/dataset_v3",
    "bc-cnn": "mypusht-train bc-cnn --dataset outputs/phase3/lerobot_dataset/dataset_v3",
    "act": "mypusht-train act --dataset outputs/phase3/lerobot_dataset/dataset_v3",
    "diffusion-policy": (
        "mypusht-train diffusion-policy --dataset outputs/phase3/lerobot_dataset/dataset_v3"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MyPushT policies. Use `mypusht-train <policy> --help` for policy options."
    )
    parser.add_argument("policy", choices=TRAIN_MODULES.keys())
    parser.add_argument("policy_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if any(item in {"-h", "--help"} for item in args.policy_args):
        print(f"usage: {POLICY_USAGE[args.policy]} [training options]")
        print()
        print("Common options:")
        print("  --dataset PATH        LeRobotDataset directory")
        print("  --out PATH            Checkpoint output path")
        print("  --steps N             Number of optimization steps")
        print("  --batch-size N        Batch size")
        print("  --device auto|cpu|cuda")
        print("  --cache-dir PATH      Optional tensor cache directory")
        print()
        print("Install the project environment to see and run the full trainer arguments.")
        return

    module_name = TRAIN_MODULES[args.policy]
    module = __import__(module_name, fromlist=["main"])

    old_argv = sys.argv
    sys.argv = [f"mypusht-train {args.policy}", *args.policy_args]
    try:
        module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
