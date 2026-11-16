"""Full20 T3 training with physical field loss and epoch-boundary exact resume."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

from adasolver.inference.plasticity2d_contract import (
    DEFAULT_CONFIG, METRIC_ID, load_config, load_data, object_hash, write_json,
)
from adasolver.metrics.field import relative_l2
from adasolver.models.plasticity2d import PlasticityModelSpec
from adasolver.models.checkpoint import sha256_file


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(sample_rng, shuffle_rng, device):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
        "sample": sample_rng.get_state(), "shuffle": shuffle_rng.get_state(),
    }


def restore_rng(state, sample_rng, shuffle_rng, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
    sample_rng.set_state(state["sample"])
    shuffle_rng.set_state(state["shuffle"])


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def validate(model, tensors, batch_size):
    model.eval()
    pos, fx, targets = tensors
    errors, frame_errors, frame_energy = [], [], []
    for start in range(0, len(pos), batch_size):
        target = targets[start:start + batch_size]
        pred = model(pos[start:start + batch_size], fx[start:start + batch_size])
        errors.append(relative_l2(pred, target))
        difference = (pred - target).reshape(len(pred), -1, 20, 2)
        frame_errors.append(difference.square().sum(dim=(1, 3)))
        frame_energy.append(target.reshape(len(pred), -1, 20, 2).square().sum(dim=(1, 3)))
    values = torch.cat(errors)
    if not torch.isfinite(values).all():
        raise RuntimeError("nonfinite full20 validation")
    return {
        "full_field_relative_l2": float(values.mean()),
        "per_sample_relative_l2": values.cpu().tolist(),
        "per_frame_squared_error_mean": torch.cat(frame_errors).mean(0).cpu().tolist(),
        "per_frame_target_energy_mean": torch.cat(frame_energy).mean(0).cpu().tolist(),
    }


def train(config, output, *, device="cuda:0", mode="variable", resume=False,
          max_epochs_this_run=None, smoke=False, arrays=None):
    cfg = copy.deepcopy(config)
    if mode not in ("variable", "fixed"):
        raise ValueError("mode must be variable or fixed")
    if smoke:
        cfg["splits"]["train"] = [0, 2]
        cfg["splits"]["validation"] = [800, 802]
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 2
    recipe = cfg["training"]
    epochs, batch_size = recipe["epochs"], recipe["batch_size"]
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs/batch_size must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and not resume:
        raise FileExistsError(f"existing training run: {output}; use --resume")
    config_id = object_hash({"config": cfg, "mode": mode, "smoke": smoke})
    seed_all(recipe["seed"])
    sample_rng = torch.Generator().manual_seed(recipe["seed"] + 100)
    shuffle_rng = torch.Generator().manual_seed(recipe["seed"] + 200)
    norm_path = output / "input_normalizer.npz"
    data = load_data(cfg, norm_path if resume else None, arrays)
    if not resume:
        np.savez(norm_path, mean=data.input_normalizer.mean.numpy(), std=data.input_normalizer.std.numpy())
    spec = PlasticityModelSpec(**cfg["model"])
    model = spec.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"],
                                 weight_decay=recipe["weight_decay"])
    ntrain = len(data.train.sample_ids)
    steps = (ntrain + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=recipe["learning_rate"], epochs=epochs,
        steps_per_epoch=steps, pct_start=recipe["pct_start"],
    )
    start_epoch, best, best_epoch = 0, float("inf"), None
    if resume:
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        if state["config_id"] != config_id:
            raise ValueError("resume config differs from original run")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best, best_epoch = state["next_epoch"], state["best"], state["best_epoch"]
        restore_rng(state["rng"], sample_rng, shuffle_rng, device)
    train_tensors = tuple(t.to(device) for t in (data.train.pos, data.train.fx, data.train.target))
    val = data.splits["validation"]
    val_tensors = tuple(t.to(device) for t in (val.pos, val.fx, val.target))
    nmin = round(3131 * recipe["nmin_fraction"]) if mode == "variable" else 3131
    code_root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), code_root / "adasolver/data/plasticity2d.py",
               code_root / "adasolver/models/transolver3.py", code_root / "adasolver/models/adapter.py",
               code_root / "adasolver/metrics/field.py", code_root / "adasolver/inference/plasticity2d_contract.py"]
    metadata = {
        "benchmark": "plasticity2d", "task_version": "full20_v1", "variant": "official_full20",
        "status": "smoke" if smoke else "running", "role": "official_operator" if mode == "variable" else "fixed_n_reference",
        "model": spec.__dict__, "nfull": 3131, "frame_start": 0, "predicted_frames": 20,
        "displacement_channels": 2, "target_layout": data.layout.order,
        "temporal_mode": "single_forward_all_frames", "metric_id": METRIC_ID,
        "nmin_fraction": recipe["nmin_fraction"] if mode == "variable" else 1., "nmin_count": nmin,
        "epochs_requested": epochs, "epochs_completed": start_epoch, "seed": recipe["seed"],
        "effective_config": cfg, "config_id": config_id, "mode": mode,
        "splits": cfg["splits"], "sample_ids": {k: list(v.sample_ids) for k, v in data.splits.items()},
        "data_path": cfg["data_path"], "data_sha256": cfg["data_sha256"],
        "input_normalizer": str(norm_path), "input_normalizer_sha256": sha256_file(norm_path),
        "target_normalization": "none", "loss": METRIC_ID, "optimizer": "AdamW", "scheduler": "OneCycleLR",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(output / "best.pt"), "last_checkpoint": str(output / "last.pt"),
        "resume_checkpoint": str(output / "resume.pt"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
                        "threads": torch.get_num_threads()},
        "command": sys.argv, "pid": os.getpid(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip(),
        "source_sha256": {str(p.relative_to(code_root)): sha256_file(p) for p in sources},
        "test_history": "last80 used in historical experiments; no new test-based selection",
    }
    write_json(output / "manifest.json", metadata)
    stop_epoch = epochs if max_epochs_this_run is None else min(epochs, start_epoch + max_epochs_this_run)
    if stop_epoch <= start_epoch:
        raise ValueError("no epochs left to run")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with (output / "history.jsonl").open("a" if resume else "w", buffering=1) as history:
        for epoch in range(start_epoch, stop_epoch):
            synchronize(device)
            started = time.perf_counter()
            model.train()
            order = torch.randperm(ntrain, generator=shuffle_rng)
            total = torch.zeros((), device=device)
            point_counts = []
            for rows_cpu in order.split(batch_size):
                count = (int(torch.randint(nmin, 3132, (1,), generator=sample_rng))
                         if mode == "variable" else 3131)
                selected = torch.randperm(3131, generator=sample_rng)[:count].to(device)
                rows = rows_cpu.to(device)
                pos, fx, target = (t[rows[:, None], selected[None, :]] for t in train_tensors)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(pos, fx)
                loss = relative_l2(prediction, target).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                total += loss.detach() * len(rows)
                point_counts.append(count)
            metrics = validate(model, val_tensors, batch_size)
            value = metrics["full_field_relative_l2"]
            if value < best:
                best, best_epoch = value, epoch
                atomic_torch_save(model.state_dict(), output / "best.pt")
            atomic_torch_save(model.state_dict(), output / "last.pt")
            atomic_torch_save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "next_epoch": epoch + 1,
                "global_step": (epoch + 1) * steps, "best": best, "best_epoch": best_epoch,
                "config_id": config_id, "rng": rng_state(sample_rng, shuffle_rng, device),
            }, output / "resume.pt")
            synchronize(device)
            row = {"epoch": epoch, "train_full_field_relative_l2": float(total / ntrain),
                   "validation": metrics, "best_validation": best, "best_epoch": best_epoch,
                   "nmin_observed": min(point_counts), "nmax_observed": max(point_counts),
                   "point_count_mean": float(np.mean(point_counts)),
                   "seconds": time.perf_counter() - started,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            metadata.update(epochs_completed=epoch + 1, best_epoch=best_epoch,
                            best_validation_full_field_relative_l2=best,
                            checkpoint_sha256=sha256_file(output / "best.pt"),
                            last_checkpoint_sha256=sha256_file(output / "last.pt"))
            metadata["status"] = "smoke" if smoke else ("trained" if epoch + 1 == epochs else "running")
            write_json(output / "manifest.json", metadata)
            print(json.dumps({k: v for k, v in row.items() if k != "validation"}
                             | {"validation_full_field_relative_l2": value}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("variable", "fixed"), default="variable")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs-this-run", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    train(load_config(args.config), args.output, device=args.device, mode=args.mode,
          resume=args.resume, max_epochs_this_run=args.max_epochs_this_run, smoke=args.smoke)


if __name__ == "__main__":
    main()
