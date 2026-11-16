"""Distributed fixed-cap Transolver-3 training for DrivAerML pressure."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

DRIVERML_CODE = Path("/mnt/ai-material-data/shuaibinqu/DrivernetML/code")
if str(DRIVERML_CODE) not in sys.path:
    sys.path.insert(0, str(DRIVERML_CODE))

from model_t3 import build_model  # noqa: E402
from train_t3_drivaerml_10k import CFD, DrivAerML10k, collate  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CFD / "data/drivaerml_surface_pt")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=3_221_250)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--slice-num", type=int, default=32)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--use-checkpoint", action="store_true")
    return parser.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        raise RuntimeError("launch with torchrun so RANK and LOCAL_RANK are defined")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def reduce_pair(total: float, count: int, device: torch.device):
    values = torch.tensor([total, count], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values[0].item(), int(values[1].item())


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)

    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if not 0 <= args.start_epoch < args.epochs:
        raise ValueError("start_epoch must satisfy 0 <= start_epoch < epochs")

    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    with args.stats.open() as stream:
        stats = json.load(stream)
    pressure_mean = float(stats["pressure_mean"])
    pressure_std = float(stats["pressure_std"])

    train_data = DrivAerML10k(
        args.root, "train", stats, args.n_points, args.seed, args.limit_train
    )
    val_data = DrivAerML10k(
        args.root, "val", stats, args.n_points, args.seed, args.limit_val,
        eval_fixed=True,
    )
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed, drop_last=True,
    )
    train_loader = DataLoader(
        train_data, batch_size=1, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    local_val = Subset(val_data, list(range(rank, len(val_data), world_size)))
    val_loader = DataLoader(
        local_val, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        dim=args.dim, n_layers=args.n_layers, heads=args.heads,
        slice_num=args.slice_num, mlp_ratio=args.mlp_ratio,
        space_dim=6, out_dim=1,
    ).to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = len(train_loader)
    remaining_epochs = args.epochs - args.start_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=steps_per_epoch * remaining_epochs,
        final_div_factor=1000.0,
    )

    best = float("inf")
    start_epoch = args.start_epoch
    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.module.load_state_dict(state["model"], strict=True)
        best = float(state.get("val_rel_l2", best))
    elif args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best_val_rel_l2"])
        if int(state["target_epochs"]) != args.epochs:
            raise ValueError("resume target epoch horizon does not match --epochs")

    log = None
    if rank == 0:
        log = (args.out_dir / "train_log.jsonl").open("a")
        print(
            f"[data] train={len(train_data)} val={len(val_data)} "
            f"n_points={args.n_points} world_size={world_size} global_bs={world_size}",
            flush=True,
        )
        print(
            f"[model] params={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M "
            f"start_epoch={start_epoch} target_epochs={args.epochs} "
            f"steps_per_epoch={steps_per_epoch}",
            flush=True,
        )

    @torch.no_grad()
    def evaluate():
        model.eval()
        rel_sum = 0.0
        count = 0
        for batch in val_loader:
            features = batch["feat"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(features, chunk_size=args.chunk_size).squeeze(-1).float()
            prediction_phys = prediction.cpu() * pressure_std + pressure_mean
            truth = batch["p_phys"]
            rel_sum += (
                torch.norm(prediction_phys - truth) / (torch.norm(truth) + 1e-12)
            ).item()
            count += 1
        rel_sum, count = reduce_pair(rel_sum, count, device)
        return rel_sum / count

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        started = time.time()
        loss_sum = 0.0
        batch_count = 0
        for batch in train_loader:
            features = batch["feat"].to(device, non_blocking=True)
            target = batch["p_n"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(
                    features, chunk_size=args.chunk_size,
                    use_checkpoint=args.use_checkpoint,
                ).squeeze(-1)
                loss = F.mse_loss(prediction.float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            batch_count += 1

        loss_sum, batch_count = reduce_pair(loss_sum, batch_count, device)
        do_validation = epoch % args.val_every == 0 or epoch == args.epochs - 1
        val_rel_l2 = evaluate() if do_validation else float("nan")
        improved = do_validation and val_rel_l2 <= best
        if improved:
            best = val_rel_l2

        if rank == 0:
            record = {
                "epoch": epoch,
                "train_loss": loss_sum / batch_count,
                "val_rel_l2": val_rel_l2,
                "lr": optimizer.param_groups[0]["lr"],
                "time_s": round(time.time() - started, 1),
                "world_size": world_size,
                "global_batch_size": world_size,
            }
            print(
                f"[ep {epoch:03d}] train_loss={record['train_loss']:.5f} "
                f"val_rel_l2={val_rel_l2:.5f} lr={record['lr']:.2e} "
                f"t={record['time_s']}s",
                flush=True,
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            checkpoint = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_rel_l2": best,
                "val_rel_l2": val_rel_l2,
                "target_epochs": args.epochs,
                "world_size": world_size,
                "global_batch_size": world_size,
                "continuation_start_epoch": args.start_epoch,
                "args": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
            }
            if improved:
                torch.save(checkpoint, args.out_dir / "checkpoint_best.pt")
            torch.save(checkpoint, args.out_dir / "checkpoint_last.pt")

    if rank == 0:
        print(f"[done] best val_rel_l2={best:.5f} -> {args.out_dir}", flush=True)
        log.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
