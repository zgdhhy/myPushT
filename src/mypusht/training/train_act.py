import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from mypusht.data.sequence_dataset import PushTSequenceDataset
from mypusht.policies.act import ACT


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


# 计算状态和动作的均值和标准差，用于数据归一化
def compute_stats(loader, device):
    states = []
    actions = []
    for batch in tqdm(loader, desc="computing stats", unit="batch"):
        states.append(batch["states"])
        actions.append(batch["actions"])
    state = torch.cat(states, dim=0).flatten(0, 1).to(device)
    action = torch.cat(actions, dim=0).flatten(0, 1).to(device)

    state_mean = state.mean(dim=0)
    state_std = state.std(dim=0).clamp_min(1e-6)
    action_mean = action.mean(dim=0)
    action_std = action.std(dim=0).clamp_min(1e-6)
    return state_mean, state_std, action_mean, action_std


def evaluate(model, loader, stats, device):
    model.eval()
    state_mean, state_std, action_mean, action_std = stats
    losses = []
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            images = batch["images"].to(device)
            actions = batch["actions"].to(device)

            states_norm = (states - state_mean) / state_std
            actions_norm = (actions - action_mean) / action_std
            loss, _, _ = model.compute_loss(states_norm, images, actions_norm)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(model, steps, train_iter, valid_iter, optimizer, stats, log_freq, device, wandb_run=None):
    step = 0
    model.train()
    state_mean, state_std, action_mean, action_std = stats
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    with tqdm(total=steps, desc="training ACT", unit="step") as pbar:
        while step < steps:
            for batch in train_iter:
                states = batch["states"].to(device)
                images = batch["images"].to(device)
                actions = batch["actions"].to(device)

                states_norm = (states - state_mean) / state_std
                actions_norm = (actions - action_mean) / action_std

                loss, action_loss, kl_loss = model.compute_loss(states_norm, images, actions_norm)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % log_freq == 0:
                    val_loss = evaluate(model, valid_iter, stats, device)
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        train_loss=f"{loss.item():.4f}",
                        val_loss=f"{val_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                    )
                    tqdm.write(
                        f"step={step:05d} train_loss={loss.item():.6f} "
                        f"val_loss={val_loss:.6f} action_loss={action_loss.item():.6f} "
                        f"kl={kl_loss.item():.6f} lr={current_lr:.2e}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": loss.item(),
                                "train/action_loss": action_loss.item(),
                                "train/kl_loss": kl_loss.item(),
                                "val/loss": val_loss,
                                "train/lr": current_lr,
                            },
                            step=step,
                        )

                step += 1
                pbar.update(1)
                if step >= steps:
                    break

def parse_args():
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="outputs/models/act.pt")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--log-freq", type=int, default=200)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mypusht")
    parser.add_argument("--wandb-name", type=str, default="act_train_01")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    # 数据集参数和模型参数
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default="outputs/cache/act")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--image-lru-size", type=int, default=256)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print("device:", device)

    dataset = PushTSequenceDataset(
        args.dataset,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        max_items=args.max_items,
        cache_lowdim=not args.no_preload,
        cache_images=not args.no_preload,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        image_lru_size=args.image_lru_size,
    )
    if len(dataset) < 10:
        raise SystemExit("Dataset too small for train/val split")

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    stats_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    sample = dataset[0]
    state_dim = sample["states"].shape[-1]
    action_dim = sample["actions"].shape[-1]
    obs_horizon = sample["states"].shape[0]
    action_horizon = sample["actions"].shape[0]
    num_cams = sample["images"].shape[1]

    stats = compute_stats(stats_loader, device)
    state_mean, state_std, action_mean, action_std = stats
    
    model = ACT(
        state_dim=state_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        num_cams=num_cams,
        kl_weight=args.kl_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "dataset": str(args.dataset),
                "train_sequences": train_size,
                "val_sequences": val_size,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "obs_horizon": obs_horizon,
                "action_horizon": action_horizon,
                "num_cams": num_cams,
                "kl_weight": args.kl_weight,
                "model_parameters": sum(p.numel() for p in model.parameters()),
            },
        )

    print("start training ACT")
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    print(
        f"train sequences={train_size} val sequences={val_size} "
        f"state_dim={state_dim} action_dim={action_dim} "
        f"obs_horizon={obs_horizon} action_horizon={action_horizon} num_cams={num_cams}"
    )
    
    train(
        model=model,
        steps=args.steps,
        train_iter=train_loader,
        valid_iter=val_loader,
        optimizer=optimizer,
        stats=stats,
        log_freq=args.log_freq,
        device=device,
        wandb_run=wandb_run,
    )
    
    outPath = Path(args.out)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": "act",
            "state_dict": model.state_dict(),
            "kwargs": model.checkpoint_kwargs(),
            "state_mean": state_mean.cpu(),
            "state_std": state_std.cpu(),
            "action_mean": action_mean.cpu(),
            "action_std": action_std.cpu(),
        },
        outPath,
    )
    print("saved:", outPath)
    if wandb_run is not None:
        wandb_run.summary["checkpoint"] = str(outPath)
        wandb_run.finish()


if __name__ == "__main__":
    main()
