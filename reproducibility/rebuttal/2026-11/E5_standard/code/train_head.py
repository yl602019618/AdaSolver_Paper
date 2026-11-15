"""Train a complete CA head for each independently trained backbone."""
import argparse
import fcntl
import os
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("elasticity", "pipe", "darcy"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    from common import ROOT, configuration, digest, read, write
    cfg = configuration(args.benchmark)
    task_lock = (ROOT / "runtime" / f"{args.benchmark}_seed{args.seed}_head.job.lock").open("a")
    fcntl.flock(task_lock, fcntl.LOCK_EX)
    if (ROOT / "checkpoints" / args.benchmark / f"seed{args.seed}" / "head/manifest.json").exists():
        return
    use_cuda = cfg["head"].get("device", "cuda") != "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if use_cuda else ""
    if use_cuda:
        lock = open(f"/tmp/adasolver_profile_gpu_{args.gpu}.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
    from learning import first_features, load_backbone, load_split
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from adasolver.inference.ca.head import TransolverGradientHead
    from adasolver.inference.ca.elasticity_stress import ElasticityStressHead
    from adasolver.train.pipe import variable_point_subset
    from adasolver.utils import seed_everything

    torch.set_num_threads(4 if use_cuda else 8)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    epochs = cfg["head"]["epochs"]
    output = ROOT / "checkpoints" / args.benchmark / f"seed{args.seed}" / "head"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        return
    began = time.perf_counter()
    point_rng = seed_everything(args.seed)
    loader_rng = torch.Generator().manual_seed(args.seed)
    model, operator_meta = load_backbone(args.benchmark, args.seed, device)
    sparse = args.benchmark != "pipe"
    data_meta = {}
    if sparse:
        from head_banks import build, tensors
        bank, data_meta = build(args.benchmark, args.seed, model, operator_meta, device)
        if args.benchmark == "elasticity":
            normalizer = np.load(ROOT / "data/elasticity/normalizer.npz")
            data_meta.update(target_mean=float(normalizer["mean"].item()),
                             target_std=float(normalizer["std"].item()))
        x, y, w = tensors(bank, data_meta, "train", device)
        xv, yv, wv = tensors(bank, data_meta, "validation", device)
        if args.benchmark == "darcy":
            train_loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True, generator=loader_rng)
            val_loader = DataLoader(TensorDataset(xv, yv), batch_size=4, shuffle=False)
    else:
        from prepare_pipe_targets import prepare
        prepare()
        data_meta = read(ROOT / "data/pipe/gradient_manifest.json")
        def dataset(split):
            d = load_split("pipe", split, device)
            raw = np.load(ROOT / "data/pipe" / f"{split}_gradient.npy")
            target = (np.log1p(raw) - data_meta["target_mean"]) / data_meta["target_std"]
            return TensorDataset(d["pos"], torch.as_tensor(target[..., None], device=device))
        train_loader = DataLoader(dataset("train"), batch_size=4, shuffle=True, generator=loader_rng)
        val_loader = DataLoader(dataset("validation"), batch_size=4, shuffle=False)
    torch.manual_seed(args.seed)
    head = (ElasticityStressHead(4) if args.benchmark == "elasticity" else TransolverGradientHead()).to(device)
    if args.benchmark == "elasticity":
        count = w.sum()
        mean = (x * w).sum(0) / count
        std = (((x - mean).square() * w).sum(0) / count).sqrt().clamp_min(1e-5)
        head.feature_mean.copy_(mean)
        head.feature_std.copy_(std)
        steps = (len(x) + 65535) // 65536
    else:
        steps = len(train_loader)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.00001, fused=use_cuda)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
                 if args.benchmark == "elasticity" else
                 torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001, epochs=epochs,
                                                     steps_per_epoch=steps, pct_start=.3))
    start, best, best_epoch, previous_seconds = 0, float("inf"), None, 0.
    resume_path = output / "resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        assert state["operator_sha256"] == operator_meta["best_sha256"]
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        point_rng.set_state(state["point_rng"].cpu())
        loader_rng.set_state(state["loader_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if use_cuda:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        start, best, best_epoch = state["epochs_completed"], state["best"], state["best_epoch"]
        previous_seconds = state["elapsed_seconds"]
    print("START HEAD", args.benchmark, args.seed, epochs, "resume", start, flush=True)
    for epoch in range(start, epochs):
        epoch_start = time.perf_counter()
        head.train()
        total = 0.
        if args.benchmark == "elasticity":
            for indices in torch.randperm(len(x), device=device).split(65536):
                element = (head(x[indices]) - y[indices]).square()
                loss = (element * w[indices]).sum() / w[indices].sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., foreach=True)
                optimizer.step()
                total += float((element * w[indices]).sum().detach())
            scheduler.step()
            head.eval()
            with torch.no_grad():
                validation = sum(float(((head(xv[i:i+65536]) - yv[i:i+65536]).square() * wv[i:i+65536]).sum())
                                 for i in range(0, len(xv), 65536)) / float(wv.sum())
            total /= float(w.sum())
        else:
            for inputs, target in train_loader:
                if args.benchmark == "pipe":
                    points = variable_point_subset(16641, 1664, point_rng).to(device)
                    pos, target = inputs[:, points], target[:, points]
                    features = first_features(model, pos, pos)
                else:
                    features = inputs
                prediction = head(features)
                loss = F.smooth_l1_loss(prediction, target, beta=1.)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), .1, foreach=True)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * len(inputs)
            total /= len(train_loader.dataset)
            head.eval()
            with torch.no_grad():
                losses = []
                for inputs, target in val_loader:
                    features = first_features(model, inputs, inputs) if args.benchmark == "pipe" else inputs
                    prediction = head(features)
                    if args.benchmark == "pipe":
                        p = prediction * data_meta["target_std"] + data_meta["target_mean"]
                        t = target * data_meta["target_std"] + data_meta["target_mean"]
                        losses.append(torch.linalg.vector_norm((p-t).flatten(1), dim=1) /
                                      torch.linalg.vector_norm(t.flatten(1), dim=1).clamp_min(1e-12))
                    else:
                        losses.append(torch.linalg.vector_norm((prediction-target).flatten(1), dim=1))
                validation = float(torch.cat(losses).mean())
        if not np.isfinite(validation):
            raise RuntimeError("Nonfinite head validation loss")
        if validation < best:
            best, best_epoch = validation, epoch + 1
            torch.save(head.state_dict(), output / "best.pt")
        row = {"benchmark": args.benchmark, "training_seed": args.seed, "head_seed": args.seed,
               "epoch": epoch + 1, "epochs": epochs, "train_loss": total, "validation_loss": validation,
               "best_validation_loss": best, "best_epoch": best_epoch,
               "epoch_seconds": time.perf_counter() - epoch_start,
               "elapsed_seconds": previous_seconds + time.perf_counter() - began,
               "gpu": args.gpu if use_cuda else None, "device": str(device), "pid": os.getpid(), "status": "running"}
        import json
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        write(ROOT / "runtime" / f"{args.benchmark}_seed{args.seed}_head.json", row)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("HEAD", args.benchmark, args.seed, epoch + 1, validation, "seconds", row["epoch_seconds"], flush=True)
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            temporary = output / "resume.tmp.pt"
            torch.save({"head": head.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epochs_completed": epoch + 1,
                "operator_sha256": operator_meta["best_sha256"], "best": best, "best_epoch": best_epoch,
                "elapsed_seconds": row["elapsed_seconds"], "point_rng": point_rng.get_state(),
                "loader_rng": loader_rng.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if use_cuda else None, "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate()}, temporary)
            temporary.replace(resume_path)
    torch.save(head.state_dict(), output / "last.pt")
    write(output / "manifest.json", {
        "benchmark": args.benchmark, "training_seed": args.seed, "head_seed": args.seed,
        "epochs_completed": epochs, "head_configuration": cfg["head"],
        "parameter_count": sum(p.numel() for p in head.parameters()),
        "operator_sha256": operator_meta["best_sha256"],
        "normalizer_sha256": operator_meta["normalizer_sha256"],
        "best_sha256": digest(output / "best.pt"), "last_sha256": digest(output / "last.pt"),
        "target_mean": data_meta.get("target_mean", 0.), "target_std": data_meta.get("target_std", 1.),
        "validation_ids": cfg["split"]["validation_ids"], "train_ids": cfg["split"]["train_ids"],
        "best_validation_loss": best, "best_epoch": best_epoch,
        "elapsed_seconds": row["elapsed_seconds"], "test_used_for_selection": False,
        "training_device": str(device), "precision": "float32",
        "status": "completed",
    })
    row["status"] = "completed"
    write(ROOT / "runtime" / f"{args.benchmark}_seed{args.seed}_head.json", row)
    print("COMPLETE HEAD", args.benchmark, args.seed, flush=True)


if __name__ == "__main__":
    main()
