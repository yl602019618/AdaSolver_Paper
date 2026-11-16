# Reproduce or verify the corrected Airfoil Cd/Cl package

From the repository root, validate every public number without a GPU:

```sh
python3 reproducibility/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/verify_package.py --data-dir data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117
```

GPU reruns require the exact external checkpoints, heads, normalizers and NACA arrays listed in `checkpoint_registry.csv` and `runtime_registry.json`. The recorded paths refer to the paper workspace. Mount the same assets at those paths when using these exact commands. The native source snapshot is included; no field-target policy is substituted. Use a fresh checkout or an empty reproduction `runs/` and `launches/` directory: completed runs are protected against overwrite.

Use the actual evaluation environment recorded in the run receipts: Python 3.10.12, PyTorch 2.10.0+cu128, CUDA 12.8 and NumPy 2.2.6. Set the repository location before entering the reproduction directory:

```sh
export AIRFOIL_CDCL_REPO="$PWD"
cd reproducibility/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117
```

Run the original frozen-checkpoint control first, using four available GPUs. Each shard processes disjoint original case IDs:

```python
# Run with python3 from the reproduction directory.
import os, subprocess, sys
jobs = []
for shard in range(4):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    jobs.append(subprocess.Popen([sys.executable, "run_evaluation.py", "--cohort", "control",
                                "--shard", str(shard), "--shards", "4",
                                "--physical-gpu", str(shard)], env=env))
for job in jobs:
    assert job.wait() == 0
```

```sh
python3 verify_control.py
```

Continue only if `control_verification.json` has `passed: true`. The training runner enforces this gate. The control uses the original checkpoint, not the P0 seed0 checkpoint. Then run the three P0 checkpoints on six available GPUs:

```python
import os, subprocess, sys
jobs = []
for seed in range(3):
    for shard in range(2):
        gpu = 2 * seed + shard
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        jobs.append(subprocess.Popen([sys.executable, "run_evaluation.py", "--cohort", "training",
                                    "--training-seed", str(seed), "--shard", str(shard), "--shards", "2",
                                    "--physical-gpu", str(gpu)], env=env))
for job in jobs:
    assert job.wait() == 0
```

Export only after every worker exits successfully, then run the validator after export has finished:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 build_package.py
python3 verify_package.py --data-dir "$AIRFOIL_CDCL_REPO/data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117"
```

`run_receipts/` stores the actual commands, devices, environment versions, start/end times and checkpoint bindings for this release. The original Table 1 aggregator is included as `reference/original_table1_aggregation.py`; the independent trapezoidal recomputation is in `result_tools.py` and `verify_package.py`.
