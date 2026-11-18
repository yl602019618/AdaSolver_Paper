"""Train the official variable-point Transolver3 operator for ShapeNet-Car."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from adasolver.data import ShapeNetCarAdapter
from adasolver.models import Transolver3ModelSpec
from adasolver.models.checkpoint import sha256_file
from adasolver.train.pipe import git_revision
from adasolver.utils import seed_everything

DEFAULT_DATA = "/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/mlcfd_data"
DEFAULT_NAME = "shapenet_car_transolver3_varpts_nmin040_e400_s0"


def _sample_batch(samples, nmin: int, generator: torch.Generator):
    """Sample one common K and independent subsets for a global batch."""
    if not samples:
        raise ValueError("cannot sample an empty batch")
    counts = [len(item) for item in samples]
    if len(set(counts)) != 1:
        raise ValueError("common-K policy requires equal point counts within a batch")
    full_count = counts[0]
    count = int(torch.randint(nmin, full_count + 1, (1,), generator=generator).item())
    selected = [torch.randperm(full_count, generator=generator)[:count] for _ in samples]
    return count, selected


def _local_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size < 1:
        raise ValueError("batch_size must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"global batch size {global_batch_size} must be divisible by world size {world_size}"
        )
    return global_batch_size // world_size


def _dual_field_loss(prediction, target, surface_mask, pressure_weight: float = 0.5):
    """Normalized-space velocity loss plus surface-only pressure loss."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if surface_mask.shape != prediction.shape[:-1]:
        raise ValueError("surface mask must match the prediction point dimensions")
    if not bool(surface_mask.any()):
        raise ValueError("sampled batch contains no surface points")
    velocity_loss = F.mse_loss(prediction[..., :3], target[..., :3])
    pressure_loss = F.mse_loss(
        prediction[..., 3][surface_mask], target[..., 3][surface_mask]
    )
    return velocity_loss + pressure_weight * pressure_loss, velocity_loss, pressure_loss


def _relative_l2(prediction, target, eps: float = 1e-12):
    numerator = torch.linalg.vector_norm((prediction - target).reshape(-1))
    denominator = torch.linalg.vector_norm(target.reshape(-1)).clamp_min(eps)
    return numerator / denominator


def _distributed_context(device_arg: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return distributed, rank, local_rank, world_size, torch.device("cuda", local_rank)
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return False, 0, 0, 1, device


def _model_state(model):
    return model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(
    model, samples, normalizer, device: torch.device, pressure_weight: float = 0.5,
    *, rank: int = 0, world_size: int = 1,
):
    """Return globally averaged physical-unit validation errors."""
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for item in samples[rank::world_size]:
        prediction_norm = model(
            item.pos[None].to(device), item.x_norm[None].to(device)
        )[0]
        prediction = normalizer.decode(prediction_norm)
        target = item.target_phys.to(device)
        mask = item.surface_mask.to(device)
        velocity_error = _relative_l2(prediction[..., :3], target[..., :3])
        pressure_error = _relative_l2(prediction[..., 3][mask], target[..., 3][mask])
        totals[0] += velocity_error.double()
        totals[1] += pressure_error.double()
        totals[2] += (velocity_error + pressure_weight * pressure_error).double()
        totals[3] += 1
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[3] == 0:
        raise ValueError("validation split must be non-empty")
    return tuple(float(value.cpu()) for value in totals[:3] / totals[3])


def _split_train_validation(samples, validation_fraction: float, seed: int):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if len(samples) < 2:
        raise ValueError("at least two training-pool samples are required")
    order = np.random.default_rng(seed).permutation(len(samples)).tolist()
    validation_count = max(1, int(round(len(samples) * validation_fraction)))
    validation_indices = set(order[:validation_count])
    train = [item for index, item in enumerate(samples) if index not in validation_indices]
    validation = [item for index, item in enumerate(samples) if index in validation_indices]
    return train, validation


def _rng_state(generator):
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "generator": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state, generator, local_rank: int = 0):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    generator.set_state(state["generator"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        states = state["cuda"]
        torch.cuda.set_rng_state(states[min(local_rank, len(states) - 1)].cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--normalizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/shapenet_car"))
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2, help="global batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--nmin-fraction", type=float, default=0.4)
    parser.add_argument("--pressure-weight", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--stop-after-epoch", type=int, default=None,
        help="test-only early stop while preserving the requested scheduler horizon",
    )
    args = parser.parse_args()
    if not 0 < args.nmin_fraction <= 1:
        raise ValueError("nmin_fraction must be in (0, 1]")
    if args.smoke:
        args.epochs = 1
        args.batch_size = 2
        args.train_samples = min(args.train_samples or 8, 8)
        args.validation_samples = min(args.validation_samples or 4, 4)
        args.name += "_smoke"

    distributed, rank, local_rank, world_size, device = _distributed_context(args.device)
    local_batch_size = _local_batch_size(args.batch_size, world_size)
    try:
        generator = seed_everything(args.seed)
        data = ShapeNetCarAdapter(
            args.data_path, normalizer_path=args.normalizer, config={"test_fold": 0}
        ).load()
        pool = data.train if args.train_samples is None else data.train[:args.train_samples]
        train, validation = _split_train_validation(pool, args.validation_fraction, args.split_seed)
        if args.validation_samples is not None:
            validation = validation[:args.validation_samples]
        if not train or not validation:
            raise ValueError("training and validation splits must be non-empty")

        counts = {len(item) for item in train}
        if len(counts) != 1:
            raise ValueError("common-K training currently requires equal full point counts")
        full_count = counts.pop()
        nmin = int(round(args.nmin_fraction * full_count))
        spec = Transolver3ModelSpec(
            space_dim=3, fun_dim=7, out_dim=4, hidden_dim=256,
            heads=8, layers=8, mlp_ratio=2, slices=32,
        )
        base_model = spec.build().to(device)
        data.output_normalizer.to(device)
        optimizer = torch.optim.AdamW(
            base_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        steps_per_epoch = len(train) // args.batch_size
        if steps_per_epoch < 1:
            raise ValueError("training split is smaller than global batch size")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.learning_rate, epochs=args.epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.3,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path("logs/shapenet_car/train") / f"{args.name}.log"
        if rank == 0:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        best_path = args.output_dir / f"{args.name}_best.pt"
        last_path = args.output_dir / f"{args.name}_last.pt"
        resume_path = args.output_dir / f"{args.name}_resume.pt"
        normalizer_path = args.output_dir / f"{args.name}_coef_norm.npz"
        manifest_path = args.output_dir / f"{args.name}_manifest.json"

        start_epoch = 0
        best = float("inf")
        continuation_world_size = world_size
        if args.resume is not None:
            state = torch.load(args.resume, map_location=device, weights_only=False)
            base_model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            saved_scheduler = state["scheduler"]
            saved_total_steps = saved_scheduler.get("total_steps")
            expected_total_steps = args.epochs * steps_per_epoch
            if saved_total_steps is not None and int(saved_total_steps) != expected_total_steps:
                raise ValueError(
                    "resume scheduler horizon mismatch: checkpoint has "
                    f"{saved_total_steps} steps but this run expects {expected_total_steps}; "
                    "resume with the original --epochs and global batch-size"
                )
            scheduler.load_state_dict(saved_scheduler)
            start_epoch = int(state["epoch"]) + 1
            best = float(state["best_validation_field_error"])
            distributed_rng = state.get("distributed_rng")
            rng = distributed_rng[rank] if distributed_rng and rank < len(distributed_rng) else state["rng"]
            _restore_rng_state(rng, generator, local_rank)
            if start_epoch >= args.epochs:
                raise ValueError(
                    f"resume checkpoint is already at epoch {start_epoch - 1}, "
                    f"which reaches the requested horizon of {args.epochs} epochs"
                )
            continuation_world_size = int(state.get("world_size", 1))

        model = DistributedDataParallel(base_model, device_ids=[local_rank]) if distributed else base_model
        log = log_path.open("a", encoding="utf-8") if rank == 0 else None
        try:
            def emit(message: str) -> None:
                if rank == 0:
                    print(message, flush=True)
                    log.write(message + "\n")
                    log.flush()

            emit(
                f"start name={args.name} epochs={args.epochs} train={len(train)} "
                f"validation={len(validation)} sealed_test={len(data.test)} "
                f"N=[{nmin},{full_count}] global_batch={args.batch_size} "
                f"local_batch={local_batch_size} world_size={world_size} "
                f"steps_per_epoch={steps_per_epoch} seed={args.seed} start_epoch={start_epoch}"
            )
            for epoch in range(start_epoch, args.epochs):
                model.train()
                order = torch.randperm(len(train), generator=generator).tolist()
                totals = torch.zeros(3, dtype=torch.float64, device=device)
                k_values = []
                for start in range(0, steps_per_epoch * args.batch_size, args.batch_size):
                    global_batch = [train[index] for index in order[start:start + args.batch_size]]
                    count, selections = _sample_batch(global_batch, nmin, generator)
                    k_values.append(count)
                    offset = rank * local_batch_size
                    local_samples = global_batch[offset:offset + local_batch_size]
                    local_selections = selections[offset:offset + local_batch_size]
                    positions = torch.stack([
                        item.pos[selected] for item, selected in zip(local_samples, local_selections)
                    ]).to(device)
                    features = torch.stack([
                        item.x_norm[selected] for item, selected in zip(local_samples, local_selections)
                    ]).to(device)
                    target = torch.stack([
                        item.target_norm[selected] for item, selected in zip(local_samples, local_selections)
                    ]).to(device)
                    mask = torch.stack([
                        item.surface_mask[selected] for item, selected in zip(local_samples, local_selections)
                    ]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(positions, features)
                    loss, velocity_loss, pressure_loss = _dual_field_loss(
                        prediction, target, mask, args.pressure_weight
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    totals += torch.stack((loss, velocity_loss, pressure_loss)).detach().double()

                if distributed:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                totals /= world_size
                velocity_error, pressure_error, field_error = validate(
                    base_model, validation, data.output_normalizer, device,
                    args.pressure_weight, rank=rank, world_size=world_size,
                )
                is_best = field_error < best
                if is_best:
                    best = field_error
                local_rng = _rng_state(generator)
                if distributed:
                    distributed_rng = [None] * world_size if rank == 0 else None
                    dist.gather_object(local_rng, distributed_rng, dst=0)
                else:
                    distributed_rng = [local_rng]
                if rank == 0:
                    state_dict = _model_state(model)
                    if is_best:
                        _atomic_torch_save(state_dict, best_path)
                    _atomic_torch_save(state_dict, last_path)
                    _atomic_torch_save(
                        {
                            "model": state_dict, "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "epoch": epoch,
                            "global_step": (epoch + 1) * steps_per_epoch,
                            "best_validation_field_error": best, "rng": distributed_rng[0],
                            "distributed_rng": distributed_rng, "world_size": world_size,
                            "global_batch_size": args.batch_size,
                            "local_batch_size": local_batch_size,
                        },
                        resume_path,
                    )
                    emit(
                        f"epoch={epoch} train_loss={float(totals[0]) / steps_per_epoch:.8f} "
                        f"train_velocity={float(totals[1]) / steps_per_epoch:.8f} "
                        f"train_pressure={float(totals[2]) / steps_per_epoch:.8f} "
                        f"validation_velocity={velocity_error:.8f} "
                        f"validation_pressure={pressure_error:.8f} "
                        f"validation_field={field_error:.8f} "
                        f"k_mean={float(np.mean(k_values)):.1f}"
                        + (" best" if is_best else "")
                    )
                if distributed:
                    dist.barrier()
                if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
                    emit(f"early_stop_after_epoch={epoch}")
                    break
        finally:
            if log is not None:
                log.close()

        if rank == 0:
            np.savez(
                normalizer_path,
                mean_in=data.input_mean.cpu().numpy(), std_in=data.input_std.cpu().numpy(),
                mean_out=data.output_mean.cpu().numpy(), std_out=data.output_std.cpu().numpy(),
            )
            manifest = {
                "benchmark": "shapenet_car", "status": "smoke" if args.smoke else "official_candidate",
                "architecture": "Transolver3", "model": spec.__dict__, "epochs": args.epochs,
                "nmin_fraction": args.nmin_fraction, "nmin_count": nmin, "nfull": full_count,
                "variable_n_policy": "common_K_per_batch", "seed": args.seed,
                "batch_size": args.batch_size, "global_batch_size": args.batch_size,
                "local_batch_size": local_batch_size, "drop_last": True,
                "distributed_backend": "nccl" if distributed else None,
                "world_size": world_size, "continuation_from_world_size": continuation_world_size,
                "optimizer": "AdamW", "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay, "max_grad_norm": args.max_grad_norm,
                "scheduler": "OneCycleLR", "pct_start": 0.3,
                "steps_per_epoch": steps_per_epoch,
                "total_optimization_steps": steps_per_epoch * args.epochs,
                "loss": "dual_field_masked_mse", "pressure_weight": args.pressure_weight,
                "train_pool_samples": len(pool), "train_samples": len(train),
                "validation_samples": len(validation), "sealed_test_samples": len(data.test),
                "validation_fraction": args.validation_fraction, "split_seed": args.split_seed,
                "test_fold": 0, "test_used_for_selection": False,
                "data_path": str(Path(args.data_path).resolve()),
                "best_validation_field_error": best,
                "checkpoint": str(best_path.resolve()), "checkpoint_sha256": sha256_file(best_path),
                "last_checkpoint": str(last_path.resolve()), "last_checkpoint_sha256": sha256_file(last_path),
                "resume_state": str(resume_path.resolve()), "normalizer": str(normalizer_path.resolve()),
                "normalizer_sha256": sha256_file(normalizer_path), "git_revision_at_start": git_revision(),
                "command": " ".join(__import__("sys").argv),
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "cuda": torch.version.cuda, "device": str(device)},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
