# MyPushT

MyPushT is a MuJoCo + SO-ARM100 PushT imitation-learning benchmark. It turns a learning prototype into a reproducible engineering project with a complete loop: simulation, teleoperation data collection, LeRobot-format conversion, policy training, unified evaluation, and result aggregation.

V1 focuses on Heuristic, BC-MLP, BC-CNN, ACT, and Diffusion Policy. VLA-style language-conditioned control is intentionally left as roadmap work.

## What Is Included

- MuJoCo PushT environments for a simple pusher and an SO-ARM100 benchmark setup.
- Teleoperation data collection with a controller mapping file in `configs/controller_mapping.json`.
- Raw episode to LeRobotDataset conversion. The public action contract is `delta_mocap_xy = np.diff(mocap_xy, axis=0)`.
- Policy implementations for Heuristic, BC-MLP, BC-CNN, ACT, and Diffusion Policy.
- Unified evaluation with normal and wide randomization splits.
- Reproducibility docs, MIT license, third-party notices, and a small result snapshot.

## Repository Policy

This repository is code-first. Generated artifacts are intentionally ignored by Git:

- datasets: `assets/lerobot_dataset/**`, `outputs/**/lerobot_dataset/**`
- checkpoints: `assets/model/**`, `*.pt`, `*.pth`, `*.ckpt`
- videos and raw episodes: `*.mp4`, `*.npz`
- caches: `outputs/**`, `__pycache__/`, wandb/cache files

Store large artifacts in GitHub Releases, Hugging Face Hub, or another external artifact store, then link them from the README.

## Setup

```bash
conda env create -f environment.yml
conda activate mypusht
pip install -e ".[dev,logging,teleop]"
```

For a CPU-only PyTorch install, replace the CUDA line in `environment.yml` with the appropriate PyTorch package for your machine.

## Quick Checks

```bash
mypusht-smoke-env --env simple --steps 20 --headless
mypusht-smoke-env --env so100 --steps 20 --headless
mypusht-convert --help
mypusht-train --help
mypusht-eval --help
pytest
```

## Data Collection And Conversion

Collect raw SO-ARM100 PushT demonstrations:

```bash
mypusht-collect --max_seconds 60 --preview_scale 2
```

Convert raw episodes into a LeRobot-compatible dataset:

```bash
mypusht-convert \
  --raw_dir outputs/phase3/raw_episodes \
  --out_dir outputs/phase3/lerobot_dataset/dataset_v3 \
  --fps 10
```

The conversion step validates frame counts and shapes before writing the dataset.

## Training

```bash
mypusht-train bc-mlp --dataset outputs/phase3/lerobot_dataset/dataset_v3
mypusht-train bc-cnn --dataset outputs/phase3/lerobot_dataset/dataset_v3 --cache-dir outputs/cache/bc_cnn
mypusht-train act --dataset outputs/phase3/lerobot_dataset/dataset_v3 --cache-dir outputs/cache/act
mypusht-train diffusion-policy --dataset outputs/phase3/lerobot_dataset/dataset_v3 --cache-dir outputs/cache/dp
```

Default checkpoints are written under `outputs/models/`.

## Evaluation

```bash
mypusht-eval --policy heuristic --episodes 10 --split normal --no-display
mypusht-eval --policy bc_mlp --ckpt outputs/models/bc_mlp.pt --split wide --episodes 10 --no-display
mypusht-eval --policy act --ckpt outputs/models/act.pt --split normal --episodes 10 --no-display
mypusht-eval --policy diffusion_policy --ckpt outputs/models/diffusion_policy.pt --split wide --episodes 10 --no-display
mypusht-aggregate
```

Metrics include success rate, mean steps, reward, final XY error, final yaw error, action smoothness, and mean policy inference latency.

## Result Snapshot

The current snapshot is stored in `docs/results/policy_comparison.csv`.

| policy | split | success_rate | mean_steps | mean_final_xy_error | mean_final_yaw_error | mean_inference_ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| act | normal | 0.9 | 84.8 | 0.0263 | 0.1293 | 243.822 |
| act | wide | 0.8 | 198.9 | 0.0255 | 0.0893 | 73.456 |
| bc_cnn | normal | 0.8 | 144.7 | 0.0285 | 0.2822 | 281.160 |
| bc_cnn | wide | 0.8 | 197.3 | 0.0242 | 0.1161 | 91.130 |
| bc_mlp | normal | 0.7 | 154.4 | 0.0330 | 0.1819 | 219.968 |
| bc_mlp | wide | 0.8 | 193.9 | 0.0283 | 0.2223 | 71.657 |
| diffusion_policy | normal | 0.8 | 148.2 | 0.0333 | 0.2000 | 489.112 |
| diffusion_policy | wide | 0.8 | 258.7 | 0.0524 | 0.2677 | 225.415 |

## Roadmap

- Publish a small demonstration dataset or artifact bundle outside Git.
- Add GitHub Actions once the dependency install time is acceptable.
- Add failure-case galleries and representative videos via Releases.
- Explore a VLA-style language-conditioned extension after the ACT/DP benchmark path is stable.

## 中文说明

MyPushT 是一个 MuJoCo + SO-ARM100 PushT 模仿学习 benchmark。项目目标不是简单堆模型，而是形成“环境、数据、训练、评估、分析”的完整闭环。

第一版只承诺 Heuristic、BC-MLP、BC-CNN、ACT、Diffusion Policy。数据集、模型权重、视频、缓存不进入 Git 仓库，后续通过 Release、Hugging Face Hub 或其他外部链接发布。

核心动作约定：训练和评估都使用 `delta_mocap_xy`，也就是 `np.diff(mocap_xy, axis=0)`。不要把它改成绝对 mocap 坐标或重新缩放的速度命令。

更多中文复现步骤见 `docs/README.zh-CN.md`。
