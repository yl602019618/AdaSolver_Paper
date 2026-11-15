"""Prepare the original physical-coordinate gradient targets for the new split."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path

from common import ROOT, write
import numpy as np
try:
    from pipe_gradient_targets import grad_mag_lsq
except ModuleNotFoundError:
    import importlib.util
    import os
    path = os.environ.get("ADASOLVER_GRAD_TARGET_FILE")
    if not path:
        raise RuntimeError("Set ADASOLVER_GRAD_TARGET_FILE to the authorized grad_targets.py source")
    spec = importlib.util.spec_from_file_location("pipe_gradient_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grad_mag_lsq = module.grad_mag_lsq


def one(item):
    split, sample = item
    directory = ROOT / "data/pipe"
    pos = np.load(directory / f"{split}_pos.npy", mmap_mode="r")
    target = np.load(directory / f"{split}_target.npy", mmap_mode="r")
    return grad_mag_lsq(pos[sample], target[sample], k=16, ridge_frac=1e-3, clip_pct=99.5)


def prepare():
    directory = ROOT / "data/pipe"
    with (directory / "gradient_build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "gradient_manifest.json").exists():
            return
        for split, samples in (("train", 1000), ("validation", 200)):
            array = np.lib.format.open_memmap(directory / f"{split}_gradient.npy", mode="w+",
                                              dtype=np.float32, shape=(samples, 16641))
            with ProcessPoolExecutor(max_workers=4) as pool:
                for i, value in enumerate(pool.map(one, ((split, j) for j in range(samples)), chunksize=4)):
                    array[i] = value
                    if (i + 1) % 100 == 0:
                        print("PIPE GRADIENT", split, i + 1, flush=True)
            array.flush()
        gradient = np.log1p(np.load(directory / "train_gradient.npy"))
        write(directory / "gradient_manifest.json", {
            "target": "physical_coordinate_gradient_magnitude", "k": 16,
            "ridge_fraction": .001, "clip_percentile": 99.5,
            "target_mean": float(gradient.mean()), "target_std": float(gradient.std() + 1e-8),
            "normalizer_fit": "1000 training cases only", "status": "completed",
        })


if __name__ == "__main__":
    prepare()
