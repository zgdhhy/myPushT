import argparse
from pathlib import Path

import wandb

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from mypusht.data.pusht_dataset import PushTDataset_MLP
from mypusht.policies.bc_mlp import BCMLP


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


# 计算状态和动作的均值和标准差，用于数据归一化
def compute_stats(loader, device):
    print("computing stats...")
    xs = []
    actions = []
    for batch in tqdm(loader, desc="computing stats", unit="batch"):
        xs.append(batch["x"])
        actions.append(batch["action"])
    x = torch.cat(xs, dim=0).to(device)
    action = torch.cat(actions, dim=0).to(device)

    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-6)
    action_mean = action.mean(dim=0)
    action_std = action.std(dim=0).clamp_min(1e-6)
    return x_mean, x_std, action_mean, action_std


def evaluate(model, loader, loss_fn, stats, device):
    model.eval()
    x_mean, x_std, action_mean, action_std = stats
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            x_norm = (x - x_mean) / x_std
            action_norm = (action - action_mean) / action_std
            pred = model(x_norm)
            losses.append(loss_fn(pred, action_norm).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(model, steps, train_iter, valid_iter, optimizer, loss_fn, stats, log_freq, device, wandb_run=None):
    step = 0
    model.train()
    x_mean, x_std, action_mean, action_std = stats
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    with tqdm(total=steps, desc="training BC-MLP", unit="step") as pbar:
        while step < steps:
            for batch in train_iter:
                x = batch["x"].to(device)
                action = batch["action"].to(device)

                x_norm = (x - x_mean) / x_std
                action_norm = (action - action_mean) / action_std

                pred = model(x_norm)
                loss = loss_fn(pred, action_norm)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % log_freq == 0:
                    val_loss = evaluate(model, valid_iter, loss_fn, stats, device)
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        train_loss=f"{loss.item():.6f}",
                        val_loss=f"{val_loss:.6f}",
                        lr=f"{current_lr:.2e}",
                    )
                    tqdm.write(f"step={step:05d} train_loss={loss.item():.6f} val_loss={val_loss:.6f}")
                    if wandb_run is not None:
                        wandb_run.log({
                            "train_loss": loss.item(),
                            "val_loss": val_loss,
                            "lr": current_lr,
                        }, step=step)
                step += 1
                pbar.update(1)
                if step >= steps:
                    break

def parse_args():
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="outputs/models/bc_mlp.pt")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--log-freq", type=int, default=200)
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mypusht")
    parser.add_argument("--wandb-name", type=str, default="cnn_train")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    # 数据集参数
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default="outputs/cache/bc_mlp")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print("device:", device)

    dataset = PushTDataset_MLP(
        args.dataset, 
        max_items=args.max_items,
        preload=not args.no_preload,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    if len(dataset) <= 10:
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
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False
    )
    stats_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False
    )

    sample = dataset[0]
    input_dim = sample["x"].numel()
    action_dim = sample["action"].numel()

    stats = compute_stats(stats_loader, device)
    x_mean, x_std, action_mean, action_std = stats

    model = BCMLP(input_dim=input_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    wandb_run = None
    if args.wandb:
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
                "input_dim": input_dim,
                "action_dim": action_dim,
                "model_parameters": sum(p.numel() for p in model.parameters()),
            },
        )
    
    print("start training BC-MLP")
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    print(f"train frames={train_size} val frames={val_size} input_dim={input_dim} action_dim={action_dim}")
    
    train(
        model=model,
        steps=args.steps,
        train_iter=train_loader,
        valid_iter=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        stats=stats,
        log_freq=args.log_freq,
        device=device,
        wandb_run=wandb_run
    )
    
    
    outPath = Path(args.out)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": "bc_mlp",
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "action_dim": action_dim,
            "x_mean": x_mean.cpu(),
            "x_std": x_std.cpu(),
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
