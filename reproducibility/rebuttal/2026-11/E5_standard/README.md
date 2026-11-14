# Reproducing the standard-benchmark training-seed supplement

The drivers require the authorized AdaSolver source package and the benchmark
data. Neural-operator and sampler source snapshots, datasets and large weights
are retained separately from this result package.

Set `ADASOLVER_SOURCE_ROOT` to the directory containing `adasolver/`,
`ADASOLVER_DATA_ROOT` to the directory containing the Elasticity, Pipe and Darcy
datasets, and `ADASOLVER_GRAD_TARGET_FILE` to the matching `grad_targets.py`
implementation used by the Pipe head. Use the Python environment listed in the
checkpoint manifests. Run from this reproduction directory:

```bash
python code/prepare_data.py --benchmark elasticity
python code/prepare_data.py --benchmark pipe
python code/prepare_data.py --benchmark darcy
python code/prepare_pipe_targets.py
python code/sampler_backend.py
python code/supervise.py
python code/finalize.py
python code/render_results.py
```

The supervisor admits currently idle GPUs and runs three complete backbones
and matching heads per benchmark. It preserves checkpoint recovery files.
Training stops at the specified full epoch counts; test results do not select
epochs or policies. The finalizer checks all 66,600 records and recomputes the
training/sampling statistics. It requires the saved checkpoints and point sets
when rerunning the complete integrity checks. To check the published numeric
tables without model weights or the AdaSolver package, run
`python code/recompute_tables.py --data <published E5_standard data directory>`.
