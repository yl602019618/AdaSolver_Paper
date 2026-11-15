"""Evaluate all locked cases, budgets and sampling seeds for one trained model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("elasticity", "pipe", "darcy"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    lock = open(f"/tmp/adasolver_profile_gpu_{args.gpu}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    from common import ROOT, configuration, digest, read, write
    from learning import first_features, load_backbone, load_split
    import numpy as np
    import torch
    from adasolver.inference import adaptive, adaptive_fast
    from adasolver.inference.acquisition import grad_score
    from adasolver.inference.base import SamplingResult
    from adasolver.inference.ca.head import TransolverGradientHead
    from adasolver.inference.ca.elasticity_stress import ElasticityStressHead
    from adasolver.inference.cpg.darcy import DarcyCPG
    from adasolver.inference.elasticity_contract import sample_rng
    from adasolver.inference.elasticity_density import DensityConfig, ElasticityDensity, scalar_score
    from adasolver.inference.pi import sample_pipe_pi, pipe_wall_distance, sample_darcy_pi, sample_elasticity_pi
    from adasolver.inference.uniform import sample_uniform
    from adasolver.layers import install_compensation
    from adasolver.utils.seed import make_numpy_rng
    from adasolver.utils.interpolation import reconstruct_field

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = configuration(args.benchmark)
    policies = read(ROOT / "config/frozen_methods.json")[args.benchmark]
    output = ROOT / "records" / args.benchmark / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        return
    device = torch.device("cuda:0")
    model, operator_meta = load_backbone(args.benchmark, args.seed, device)
    head_path = ROOT / "checkpoints" / args.benchmark / f"seed{args.seed}" / "head"
    head_meta = read(head_path / "manifest.json")
    assert head_meta["status"] == "completed" and head_meta["head_seed"] == args.seed
    assert head_meta["epochs_completed"] == cfg["head"]["epochs"]
    assert head_meta["operator_sha256"] == operator_meta["best_sha256"]
    assert digest(head_path / "best.pt") == head_meta["best_sha256"]
    head = (ElasticityStressHead(4) if args.benchmark == "elasticity" else TransolverGradientHead()).to(device)
    head.load_state_dict(torch.load(head_path / "best.pt", map_location=device, weights_only=True))
    head.requires_grad_(False).eval()
    context, patched = install_compensation(model)
    assert patched == 8
    data = load_split(args.benchmark, "test", device)
    coords = data["pos"].cpu().numpy()
    truth = data["target"].cpu().numpy()
    norms = np.load(ROOT / "data" / args.benchmark / "normalizer.npz")
    mean = torch.as_tensor(norms["mean"], device=device)
    std = torch.as_tensor(norms["std"], device=device)
    ids = cfg["split"]["test_ids"]
    assert len(ids) == len(truth) == 200
    nfull = cfg["nfull"]
    reference = None if args.benchmark == "elasticity" else np.asarray(
        np.load(ROOT / "data" / args.benchmark / "reference.npy"), dtype=np.float64)
    geometry = {}
    distance = {}
    coarse_cache = {}
    if args.benchmark == "pipe":
        labels, masses = adaptive.build_patches(reference, 24, seed=0)
        patch_cache = adaptive_fast.build_patch_cache(reference, labels, 24)
    elif args.benchmark == "darcy":
        cpg = DarcyCPG(reference, **policies["cpg"])
        labels, masses = cpg.patch_id, cpg.patch_mass
    if (ROOT / "config/sampler_backend.json").exists():
        from sampler_backend import install
        install()

    def rng_for(local, sampling_seed):
        sid = ids[local]
        if args.benchmark == "elasticity":
            return sample_rng(sid, sampling_seed)
        if args.benchmark == "darcy":
            return make_numpy_rng(sampling_seed, sid)
        return np.random.default_rng(1000 * sampling_seed + local)

    @torch.no_grad()
    def acquire(local, sampling_seed, method):
        key = local, sampling_seed, method
        if key not in coarse_cache:
            count = 97 if args.benchmark == "elasticity" else round(.1 * nfull)
            coarse = rng_for(local, sampling_seed).choice(nfull, count, replace=False)
            points = torch.as_tensor(coarse, device=device)
            pos = data["pos"][local:local + 1, points]
            fx = data["fx"][local:local + 1, points] if data["fx"] is not None else None
            context.clear()
            value = model(pos, fx) if method == "cpg" else head(first_features(model, pos, fx))
            if args.benchmark == "elasticity" or method == "cpg" and args.benchmark == "pipe":
                value = value * std + mean
            value = value[0].cpu().numpy()
            if method == "ca" and args.benchmark != "elasticity":
                raw = np.asarray(value, np.float64) if args.benchmark == "darcy" else value
                decoded = np.expm1(np.maximum(raw.reshape(-1) * head_meta["target_std"] + head_meta["target_mean"], 0.))
                value = decoded if args.benchmark == "darcy" else decoded ** 2
            coarse_cache[key] = coarse, value
        return coarse_cache[key]

    def select(local, sampling_seed, method, count, budget):
        rng = rng_for(local, sampling_seed)
        ref = coords[local] if reference is None else reference
        if method == "uniform":
            return sample_uniform(nfull, count, rng), None
        if method == "pi":
            if args.benchmark == "elasticity":
                selection = sample_elasticity_pi(ref, count, rng, **policies["pi"]["pi"])
            elif args.benchmark == "pipe":
                if local not in distance:
                    distance[local] = pipe_wall_distance(coords[local], (129, 129))
                selection = sample_pipe_pi(distance[local], count, rng, **policies["pi"])
            else:
                coefficients = data["fx"][local].cpu().numpy() * norms["x_std"].reshape(1, -1) + norms["x_mean"].reshape(1, -1)
                selection = sample_darcy_pi(coefficients, ref, count, rng, **policies["pi"])
            return selection, None
        coarse, field = acquire(local, sampling_seed, method)
        if args.benchmark == "elasticity":
            c = DensityConfig(**policies[method]["density"])
            if local not in geometry:
                geometry[local] = ElasticityDensity(ref)
            score = scalar_score(ref[coarse], field, c.score, c.neighbors)
            selection = geometry[local].refine(coarse, score, count, c,
                                               sample_rng(ids[local], sampling_seed, refinement=True))
        elif args.benchmark == "pipe":
            score = grad_score(ref[coarse], field, k=8) if method == "cpg" else field
            refinement_rng = np.random.default_rng(7_000_000 + 1000 * sampling_seed + local)
            chosen, info = adaptive_fast.refine_once(coarse, score, ref, labels, masses, count,
                kappa=4., seed=int(refinement_rng.integers(1 << 30)), patch_cache=patch_cache, placement="fps")
            weights = adaptive.comp_weights_for_subset(chosen, ref, nfull, beta=1.)
            selection = SamplingResult(chosen, 1. / weights.astype(float), weights, info)
        else:
            rng.choice(nfull, len(coarse), replace=False)
            if method == "cpg":
                selection = cpg.refine(coarse, field, count, rng)
            else:
                chosen, info = adaptive.refine_once(coarse, field, ref, labels, masses, count,
                    kappa=2., seed=int(rng.integers(1 << 30)))
                weights = adaptive.comp_weights_for_subset(chosen, ref, nfull, beta=1.)
                selection = SamplingResult(chosen, 1. / weights.astype(float), weights, info)
        return selection, coarse

    @torch.no_grad()
    def predict(local, selection, uniform):
        points = torch.as_tensor(selection.indices, device=device)
        weights = None if uniform else torch.as_tensor(selection.compensation, device=device)
        context.set(weights)
        pos = data["pos"][local:local + 1, points]
        fx = data["fx"][local:local + 1, points] if data["fx"] is not None else None
        value = (model(pos, fx) * std + mean)[0].cpu().numpy()
        context.clear()
        return value

    def record(local, sampling_seed, method, budget, selection, coarse, prediction, fallback):
        selection.validate(nfull, round(nfull * budget))
        if coarse is not None:
            assert len(coarse) == (97 if args.benchmark == "elasticity" else round(.1 * nfull))
            assert np.array_equal(selection.indices[:len(coarse)], coarse)
        assert np.isfinite(prediction).all()
        numerator = float(np.linalg.norm(prediction - truth[local]))
        denominator = float(np.linalg.norm(truth[local]))
        return {"benchmark": args.benchmark, "training_seed": args.seed,
                "head_seed": args.seed if method == "ca" else None,
                "sampling_seed": sampling_seed, "sample_id": ids[local], "method": method,
                "budget_fraction": budget, "nfull": nfull, "selected_count": len(selection.indices),
                "coarse_count": 0 if coarse is None else len(coarse),
                "relative_l2": numerator / max(denominator, 1e-12),
                "relative_l2_numerator": numerator, "relative_l2_denominator": denominator,
                "interpolation_fallback_count": fallback, "status": "ok",
                "operator_sha256": operator_meta["best_sha256"],
                "head_sha256": head_meta["best_sha256"] if method == "ca" else None}

    started = time.perf_counter()
    full_path = output / "full.jsonl"
    if not full_path.exists():
        rows = []
        selection = SamplingResult(np.arange(nfull), np.ones(nfull), np.ones(nfull, np.float32))
        for local in range(200):
            prediction = predict(local, selection, True)
            rows.append(record(local, None, "full", 1., selection, None, prediction, 0))
        temporary = full_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
        temporary.replace(full_path)
    condition_files = [full_path]
    for sampling_seed in (0, 1, 2):
        for budget in (.375, .5, .75):
            count = round(nfull * budget)
            stem = f"sampling{sampling_seed}_n{count}"
            path, array_path = output / (stem + ".jsonl"), output / (stem + ".npz")
            condition_files.append(path)
            if path.exists() and array_path.exists():
                continue
            arrays = {m: np.empty((200, count), np.uint16) for m in ("uniform", "pi", "cpg", "ca")}
            rows = []
            for local in range(200):
                ref = coords[local] if reference is None else reference
                for method in ("uniform", "pi", "cpg", "ca"):
                    selection, coarse = select(local, sampling_seed, method, count, budget)
                    sparse_prediction = predict(local, selection, method == "uniform")
                    prediction, fallback = reconstruct_field(selection.indices, sparse_prediction, ref)
                    rows.append(record(local, sampling_seed, method, budget, selection, coarse, prediction, fallback))
                    arrays[method][local] = selection.indices
                if (local + 1) % 25 == 0:
                    write(ROOT / "runtime" / f"{args.benchmark}_seed{args.seed}_evaluate.json", {
                        "status": "running", "benchmark": args.benchmark, "training_seed": args.seed,
                        "sampling_seed": sampling_seed, "budget": budget, "cases_completed": local + 1,
                        "gpu": args.gpu, "pid": os.getpid(), "elapsed_seconds": time.perf_counter() - started})
                    print("EVAL", args.benchmark, args.seed, sampling_seed, budget, local + 1, flush=True)
            np.savez_compressed(array_path, **arrays)
            temporary = path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
            temporary.replace(path)
    rows = [json.loads(line) for p in condition_files for line in p.read_text().splitlines() if line]
    assert len(rows) == 7400
    assert len({(r["sample_id"], r["sampling_seed"], r["method"], r["budget_fraction"]) for r in rows}) == 7400
    write(output / "completion.json", {"status": "completed", "benchmark": args.benchmark,
        "training_seed": args.seed, "head_seed": args.seed, "records": len(rows),
        "operator_sha256": operator_meta["best_sha256"], "head_sha256": head_meta["best_sha256"],
        "normalizer_sha256": operator_meta["normalizer_sha256"], "configurations": policies,
        "files": [p.name for p in condition_files], "elapsed_seconds": time.perf_counter() - started})
    write(ROOT / "runtime" / f"{args.benchmark}_seed{args.seed}_evaluate.json", {
        "status": "completed", "benchmark": args.benchmark, "training_seed": args.seed,
        "records": len(rows), "gpu": args.gpu, "pid": os.getpid()})
    print("COMPLETE EVAL", args.benchmark, args.seed, len(rows), flush=True)


if __name__ == "__main__":
    main()
