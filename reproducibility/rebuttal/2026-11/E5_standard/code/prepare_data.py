"""Prepare full benchmark arrays with validation separated from the test cohort."""
import argparse
import os
from pathlib import Path

from common import ROOT, configuration, write
import numpy as np
import scipy.io
import torch

from adasolver.data.pipe import UnitTransformer
from adasolver.utils.interpolation import index_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("elasticity", "pipe", "darcy"), required=True)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("ADASOLVER_DATA_ROOT"))
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("Supply --data-root or set ADASOLVER_DATA_ROOT")
    torch.set_num_threads(4)
    cfg = configuration(args.benchmark)
    output = ROOT / "data" / args.benchmark
    output.mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        print("Prepared data already present:", output, flush=True)
        return
    source = args.data_root / args.benchmark
    split = cfg["split"]
    arrays = {}
    if args.benchmark == "elasticity":
        source = source / "elasticity/Meshes"
        xy = np.load(source / "Random_UnitCell_XY_10.npy", mmap_mode="r")
        stress = np.load(source / "Random_UnitCell_sigma_10.npy", mmap_mode="r")
        for name in ("train", "validation", "test"):
            ids = split[name + "_ids"]
            arrays[name] = {
                "pos": np.ascontiguousarray(xy[:, :, ids].transpose(2, 0, 1), dtype=np.float32),
                "target": np.ascontiguousarray(stress[:, ids].T[:, :, None], dtype=np.float32),
            }
        reference = None
    elif args.benchmark == "pipe":
        x = np.load(source / "Pipe_X.npy", mmap_mode="r")
        y = np.load(source / "Pipe_Y.npy", mmap_mode="r")
        q = np.load(source / "Pipe_Q.npy", mmap_mode="r")
        for name in ("train", "validation", "test"):
            ids = split[name + "_ids"]
            arrays[name] = {
                "pos": np.stack((x[ids], y[ids]), axis=-1).reshape(len(ids), -1, 2).astype(np.float32),
                "target": np.array(q[ids, 0], dtype=np.float32).reshape(len(ids), -1, 1),
            }
        reference = index_grid((129, 129))
    else:
        def load(filename, ids):
            data = scipy.io.loadmat(source / filename, variable_names=["coeff", "sol"])
            return {key: np.ascontiguousarray(data[field][ids, ::5, ::5], dtype=np.float32).reshape(len(ids), -1, 1)
                    for key, field in (("fx", "coeff"), ("target", "sol"))}
        arrays["train"] = load("piececonst_r421_N1024_smooth1.mat", np.arange(1000))
        evaluation = load("piececonst_r421_N1024_smooth2.mat", np.arange(400))
        arrays["test"] = {k: v[:200] for k, v in evaluation.items()}
        arrays["validation"] = {k: v[200:400] for k, v in evaluation.items()}
        a = np.linspace(0., 1., 85)
        gx, gy = np.meshgrid(a, a)
        reference = np.column_stack((gx.ravel(), gy.ravel())).astype(np.float32)
        for values in arrays.values():
            values["pos"] = reference

    normalizer = UnitTransformer(torch.from_numpy(arrays["train"]["target"]))
    norms = {"mean": normalizer.mean.numpy(), "std": normalizer.std.numpy()}
    if args.benchmark == "darcy":
        xnorm = UnitTransformer(torch.from_numpy(arrays["train"]["fx"]))
        norms.update(x_mean=xnorm.mean.numpy(), x_std=xnorm.std.numpy())
        for values in arrays.values():
            values["fx"] = xnorm.encode(torch.from_numpy(values["fx"])).numpy()
    for name, values in arrays.items():
        for key, value in values.items():
            assert np.isfinite(value).all(), (name, key)
            np.save(output / f"{name}_{key}.npy", value, allow_pickle=False)
        encoded = normalizer.encode(torch.from_numpy(values["target"])).numpy()
        np.save(output / f"{name}_encoded.npy", encoded, allow_pickle=False)
    np.savez(output / "normalizer.npz", **norms)
    if reference is not None:
        np.save(output / "reference.npy", reference)
    write(output / "manifest.json", {
        "benchmark": args.benchmark, "nfull": cfg["nfull"],
        "samples": {name: len(values["target"]) for name, values in arrays.items()},
        "split": split, "normalizer_fit": "training samples only",
        "mean": norms["mean"].tolist(), "std": norms["std"].tolist(),
    })
    print("Prepared", args.benchmark, "1000 train / 200 validation / 200 test", flush=True)


if __name__ == "__main__":
    main()
