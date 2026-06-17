# MyPushT 中文复现说明

MyPushT 的定位是一个可复现的具身智能学习工程：基于 MuJoCo 构建 SO-ARM100 PushT 接触操作任务，并比较 Heuristic、BC-MLP、BC-CNN、ACT、Diffusion Policy。

## 1. 环境安装

```bash
conda env create -f environment.yml
conda activate mypusht
pip install -e ".[dev,logging,teleop]"
```

如果你的机器没有 CUDA，请按 PyTorch 官方说明把 `environment.yml` 里的 CUDA 依赖换成 CPU 版本。

## 2. 最小检查

```bash
mypusht-smoke-env --env simple --steps 20 --headless
mypusht-smoke-env --env so100 --steps 20 --headless
pytest
```

如果 SO100 smoke test 失败，优先检查 `assets/so100/**` 是否存在，以及 MuJoCo 能否加载 XML/STL。

## 3. 采集数据

手柄映射在 `configs/controller_mapping.json`。

```bash
mypusht-collect --max_seconds 60 --preview_scale 2
```

raw episode 默认保存到 `outputs/phase3/raw_episodes`。这些 `.npz` 文件不要提交到 Git。

## 4. 转换 LeRobot 数据集

```bash
mypusht-convert \
  --raw_dir outputs/phase3/raw_episodes \
  --out_dir outputs/phase3/lerobot_dataset/dataset_v3 \
  --fps 10
```

动作语义必须保持一致：

```python
action = np.diff(mocap_xy, axis=0).astype(np.float32)
```

也就是每一步 action 表示 mocap target 的 XY 增量，不是绝对位置。

## 5. 训练策略

```bash
mypusht-train bc-mlp --dataset outputs/phase3/lerobot_dataset/dataset_v3
mypusht-train bc-cnn --dataset outputs/phase3/lerobot_dataset/dataset_v3
mypusht-train act --dataset outputs/phase3/lerobot_dataset/dataset_v3
mypusht-train diffusion-policy --dataset outputs/phase3/lerobot_dataset/dataset_v3
```

默认模型输出到 `outputs/models/`。这些 `.pt` 权重不要提交到 Git。

## 6. 统一评估

```bash
mypusht-eval --policy heuristic --split normal --episodes 10 --no-display
mypusht-eval --policy bc_mlp --ckpt outputs/models/bc_mlp.pt --split wide --episodes 10 --no-display
mypusht-aggregate
```

评估指标包括成功率、平均步数、最终 XY 误差、最终 yaw 误差、动作平滑度、策略推理延迟。

## 7. GitHub 发布原则

- 提交代码、配置、文档、测试、小型结果表。
- 不提交 `outputs/**`、`assets/model/**`、`assets/lerobot_dataset/**`、`.pt`、`.mp4`、`.npz`。
- SO-ARM100 资源公开前要确认来源和许可证；无法确认时，README 只保留下载说明。
