"""Full Transolver3 training with validation-only checkpoint selection and resume."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=("elasticity", "pipe", "darcy"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    lock = open(f"/tmp/adasolver_profile_gpu_{args.gpu}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    from common import ROOT, configuration, digest, write
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from adasolver.models import Transolver3ModelSpec
    from adasolver.metrics import relative_l2
    from adasolver.train.pipe import variable_point_subset
    from adasolver.utils import seed_everything

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = configuration(args.benchmark)
    settings = cfg["training"]
    data_root = ROOT / "data" / args.benchmark
    output = ROOT / "checkpoints" / args.benchmark / f"seed{args.seed}" / "backbone"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        print("Backbone already complete", output, flush=True)
        return
    began = time.perf_counter()
    device = torch.device("cuda:0")
    point_rng = seed_everything(args.seed)
    loader_rng = torch.Generator().manual_seed(args.seed)
    norms = np.load(data_root / "normalizer.npz")
    mean = torch.as_tensor(norms["mean"], device=device)
    std = torch.as_tensor(norms["std"], device=device)

    def dataset(split):
        key = "encoded" if split == "train" else "target"
        target = torch.as_tensor(np.load(data_root / f"{split}_{key}.npy"), device=device)
        pos = torch.as_tensor(np.load(data_root / f"{split}_pos.npy"), device=device)
        if pos.ndim == 2:
            pos = pos[None].expand(len(target), -1, -1)
        fx = (torch.as_tensor(np.load(data_root / f"{split}_fx.npy"), device=device)
              if args.benchmark == "darcy" else pos if args.benchmark == "pipe" else pos[:, :, :0])
        return TensorDataset(pos, fx, target)

    train_data, validation_data = dataset("train"), dataset("validation")
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True, generator=loader_rng)
    validation_loader = DataLoader(validation_data, batch_size=settings["batch_size"], shuffle=False)
    model = Transolver3ModelSpec(**cfg["model"]).build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], fused=True)
    if settings["scheduler"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=settings["learning_rate"],
            epochs=settings["epochs"], steps_per_epoch=len(train_loader), pct_start=settings["pct_start"])
    start, best, best_epoch, previous_seconds = 0, float("inf"), None, 0.
    resume_path = output / "resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        point_rng.set_state(state["point_rng"].cpu())
        loader_rng.set_state(state["loader_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        start, best, best_epoch = state["epochs_completed"], state["best"], state["best_epoch"]
        previous_seconds = state["elapsed_seconds"]
        del state
    runtime_path = ROOT / "runtime" / f'{args.benchmark}_seed{args.seed}_backbone.json'
    print(f"START {args.benchmark} seed={args.seed} epochs={settings['epochs']} resume={start} "
          f"train=1000 validation=200 N=[{settings['nmin_count']},{cfg['nfull']}] gpu={args.gpu}", flush=True)

    for epoch in range(start, settings["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        total = torch.zeros((), device=device)
        for pos, fx, target in train_loader:
            selected = variable_point_subset(cfg["nfull"], settings["nmin_count"], point_rng).to(device)
            pos, target = pos[:, selected], target[:, selected]
            fx = fx[:, selected] if fx.shape[-1] else None
            prediction = model(pos, fx)
            loss = relative_l2(prediction, target).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"], foreach=True)
            optimizer.step()
            if settings["scheduler"] == "OneCycleLR":
                scheduler.step()
            total += loss.detach()
        if settings["scheduler"] == "CosineAnnealingLR":
            scheduler.step()
        model.eval()
        with torch.no_grad():
            values = []
            for pos, fx, target in validation_loader:
                prediction = model(pos, fx if fx.shape[-1] else None) * std + mean
                values.append(relative_l2(prediction, target))
            validation = float(torch.cat(values).mean())
        if not np.isfinite(validation):
            raise RuntimeError("Nonfinite validation loss")
        if validation < best:
            best, best_epoch = validation, epoch + 1
            torch.save(model.state_dict(), output / "best.pt")
        row = {"benchmark": args.benchmark, "seed": args.seed, "epoch": epoch + 1,
               "epochs": settings["epochs"], "train_relative_l2": float(total) / 1000,
               "validation_relative_l2": validation, "best_validation_relative_l2": best,
               "best_epoch": best_epoch, "epoch_seconds": time.perf_counter() - epoch_start,
               "elapsed_seconds": previous_seconds + time.perf_counter() - began,
               "gpu": args.gpu, "pid": os.getpid(), "status": "running"}
        with (output / "history.jsonl").open("a") as stream:
            import json
            stream.write(json.dumps(row) + "\n")
        write(runtime_path, row)
        print(f"epoch={epoch+1}/{settings['epochs']} train={row['train_relative_l2']:.7f} "
              f"val={validation:.7f} best={best:.7f} sec={row['epoch_seconds']:.2f}", flush=True)
        if (epoch + 1) % 10 == 0 or epoch + 1 == settings["epochs"]:
            temporary = output / "resume.tmp.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epochs_completed": epoch + 1,
                "best": best, "best_epoch": best_epoch, "elapsed_seconds": row["elapsed_seconds"],
                "point_rng": point_rng.get_state(), "loader_rng": loader_rng.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, temporary)
            temporary.replace(resume_path)
    torch.save(model.state_dict(), output / "last.pt")
    manifest = {"benchmark": args.benchmark, "training_seed": args.seed,
        "epochs_completed": settings["epochs"], "training": settings, "model": cfg["model"],
        "split": cfg["split"], "test_used_for_selection": False,
        "checkpoint_selection": settings["checkpoint_selection"], "best_epoch": best_epoch,
        "best_validation_relative_l2": best, "elapsed_seconds": row["elapsed_seconds"],
        "best_sha256": digest(output / "best.pt"), "last_sha256": digest(output / "last.pt"),
        "normalizer_sha256": digest(data_root / "normalizer.npz"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "physical_gpu": args.gpu},
        "status": "completed"}
    write(output / "manifest.json", manifest)
    row["status"] = "completed"
    write(runtime_path, row)
    print("COMPLETE", args.benchmark, args.seed, flush=True)


if __name__ == "__main__":
    main()
