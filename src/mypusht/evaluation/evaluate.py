import argparse
import csv
import json
from statistics import mean
import time
from pathlib import Path

import numpy as np

from mypusht.evaluation.config import ensure_res_dirs, EVAL_SPLITS, POLICY_SPECS


# 关于设备选择
def resolve_device(device_arg):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


# 关于视频录制
class ImageioVideoWriter:
    def __init__(self, path, fps=20):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264")

    def write(self, frame):
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.writer.append_data(np.ascontiguousarray(rgb_frame))

    def release(self):
        self.writer.close()


# 关于计算action平滑度 有点像是action的方差
def action_smoothness(actions):
    if len(actions) < 2:
        return 0.0
    action_array = np.stack(actions)
    return float(np.linalg.norm(np.diff(action_array, axis=0), axis=1).mean())


# 关于显示和评估指标
def make_display_frame(obs, episode_idx, seed, step, max_steps, reward, info, action):
    import cv2

    top = cv2.cvtColor(obs["images"]["cam_top"], cv2.COLOR_RGB2BGR)
    side = cv2.cvtColor(obs["images"]["cam_side"], cv2.COLOR_RGB2BGR)
    mocap_xy = obs["mocap_xy"]
    frame = np.concatenate([
        cv2.resize(top, (448, 448)),
        cv2.resize(side, (448, 448)),
    ], axis=1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 92), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, dst=frame)

    success_flag = int(bool(info.get("success", False)))
    xy_error = float(info.get("xy_error", np.nan))
    yaw_error = float(info.get("yaw_error", np.nan))
    ax = float(action[0]) if len(action) > 0 else np.nan
    ay = float(action[1]) if len(action) > 1 else np.nan
    mx = float(mocap_xy[0]) if len(mocap_xy) > 0 else np.nan
    my = float(mocap_xy[1]) if len(mocap_xy) > 1 else np.nan

    lines = [
        f"episode={episode_idx} seed={seed} step={step + 1}/{max_steps} success={success_flag}",
        f"return={reward:.3f} xy_error={xy_error:.4f} yaw_error={yaw_error:.3f}",
        f"mocap=({mx:.4f}, {my:.4f}) action=({ax:.4f}, {ay:.4f})",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 24 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def show_frame(window_name, frame, wait_ms=1):
    import cv2

    try:
        cv2.imshow(window_name, frame)
    except cv2.error as exc:
        raise RuntimeError(
            "cv2.imshow failed. Run with --no-display in headless environments."
        ) from exc
    key = cv2.waitKey(wait_ms) & 0xFF
    return key == ord("q")



def run_episode(env, policy, seed, max_steps, video_path=None,
                episode_idx=None, display=True, window_name="eval_policy"):
    obs, _ = env.reset(seed=seed)
    policy.reset_episode()
    writer = ImageioVideoWriter(video_path) if video_path is not None else None
    actions = []
    inference_times = []
    terminated = False
    truncated = False
    start_time = time.perf_counter()
    step = -1

    try:
        for step in range(max_steps):
            predict_start = time.perf_counter()
            action = policy.predict(obs)
            inference_times.append(time.perf_counter() - predict_start)
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)

            if writer is not None or display:
                frame = make_display_frame(
                    obs=obs, episode_idx=episode_idx, seed=seed,
                    step=step, max_steps=max_steps, reward=reward,
                    info=info, action=action,
                )
            if writer is not None:
                writer.write(frame)
            if display:
                if show_frame(window_name, frame):
                    truncated = True
                    break
            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.release()

    elapsed_sec = time.perf_counter() - start_time

    res = {
        "seed": seed,
        "success": int(bool(info.get("success", terminated))),
        "truncated": int(bool(truncated)),
        "steps": step + 1,
        "reward": round(reward, 4),
        "final_xy_error": round(float(info.get('xy_error', np.nan)), 4),
        "final_yaw_error": round(float(info.get('yaw_error', np.nan)), 4),
        "action_smoothness": round(action_smoothness(actions), 4),
        "elapsed_sec": round(elapsed_sec, 4),
        "inference_ms_mean": round(mean(inference_times) * 1000, 4) if inference_times else float("nan"),
    }
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["heuristic", "bc_mlp", "bc_cnn", "act", "diffusion_policy"], required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--split", choices=list(EVAL_SPLITS.keys()), default="normal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--window-name", type=str, default="eval_<policy>")
    return parser.parse_args()


def main():
    args = parse_args()
    from mypusht.evaluation.adapters import load_env, load_policy

    policy_meta = POLICY_SPECS[args.policy]
    ckpt = args.ckpt if args.ckpt is not None else policy_meta.ckpt
    split = EVAL_SPLITS[args.split]
    seed_start = args.seed if args.seed is not None else split.seed_start
    episodes = args.episodes if args.episodes is not None else split.episodes
    max_steps = args.max_steps if args.max_steps is not None else split.max_steps
    device = resolve_device(args.device)

    res_paths = ensure_res_dirs()

    print("policy:", args.policy)
    print("device:", device)
    print("episodes:", episodes)

    policy = load_policy(args.policy, ckpt=ckpt, device=device)
    env = load_env(args.split, max_steps=split.max_steps)

    window_name = args.window_name.replace("<policy>", args.policy)

    rows = []
    try:
        for episode_idx in range(episodes):
            seed = seed_start + episode_idx
            video_path = res_paths['videos'] / f"{args.policy}_seed_{seed}.mp4" \
                if args.save_video else None

            result = run_episode(
                env=env, policy=policy, seed=seed,
                max_steps=max_steps, video_path=video_path,
                episode_idx=episode_idx,
                display=not args.no_display,
                window_name=window_name
            )
            result["policy"] = args.policy
            result["split"] = args.split
            result["video_path"] = str(video_path) if video_path is not None else ""
            rows.append(result)

            print(
                f"[{args.policy}][{args.split}] "
                f"episode {episode_idx + 1}/{episodes} seed={seed} "
                f"success={result['success']} "
                f"steps={result['steps']} "
                f"xy={result['final_xy_error']:.4f} "
                f"yaw={result['final_yaw_error']:.4f} ",
                flush=True,
            )
    finally:
        env.close()
        if not args.no_display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if not rows:
        raise SystemExit("No evaluation episodes were run.")
    
    result_path = res_paths["results"] / f"{args.policy}_{args.split}.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "policy": policy_meta.display_name,
        "family": policy_meta.family,
        "action_mode": policy_meta.action_mode,
        "split": args.split,
        "episodes": episodes,
        "success_rate": mean([row["success"] for row in rows]),
        "mean_steps": mean([row["steps"] for row in rows]),
        "mean_xy_error": mean([row["final_xy_error"] for row in rows]),
        "mean_yaw_error": mean([row["final_yaw_error"] for row in rows]),
        "mean_action_smoothness": mean([row["action_smoothness"] for row in rows]),
        "mean_inference_ms": mean([row["inference_ms_mean"] for row in rows]),
    }

    summary_path = res_paths["results"] / f"{args.policy}_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nsummary")
    print("policy:", args.policy)
    print("success_rate:", round(summary["success_rate"], 3))
    print("mean_steps:", round(summary["mean_steps"], 3))
    print("mean_xy_error:", round(summary["mean_xy_error"], 3))
    print("mean_yaw_error:", round(summary["mean_yaw_error"], 3))

    print("\nSaved:")
    print(result_path)
    print(summary_path)

if __name__ == "__main__":
    main()
