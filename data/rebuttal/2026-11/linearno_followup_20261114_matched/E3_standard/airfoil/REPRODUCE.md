# Reproduce the locked Airfoil comparison

Run from this directory. The data path and the train-only normalizer are recorded in `config/data_manifest.json`; the shared IDs are in `../split_manifest.json`. Native source files and their hashes are in `code/vendor/` and `config/source_manifest.json`. Training uses Python 3.10, PyTorch 2.10.0+cu128 and the dependency versions recorded with timing setup.

The current persistent launch commands are in `runtime/launches.json`. Equivalent first-run commands in a fresh copy of this output directory are:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 taskset -c 16-23 python3 -u code/workflow.py --architecture t3
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 taskset -c 24-31 python3 -u code/workflow.py --architecture linearno
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4 taskset -c 32-39 python3 -u code/workflow.py --architecture linearno_conv
```

These are long independent workflows: 1,000-epoch backbone training, preliminary Full/Uniform/PI/CPG evaluation, own-feature CA head training for the point models, held-out reconstruction acceptance and the complete test matrix. The conv-temp workflow skips all subsampling and head stages. The scripts preserve existing outputs and require explicit resume handling for partial training (`code/train.py --architecture NAME --resume`). No test criterion selects a checkpoint.

The reconstruction gate must match the accepted implementation version and hash in `config/evaluation_gate.json`. Each evaluation stores the source that actually ran, its original/optimized full-field metrics, prediction arrays and selected/coarse index hashes. Local statistical aggregation is:

```bash
python3 code/aggregate.py
python3 code/validate_accuracy.py --architecture t3
python3 code/validate_accuracy.py --architecture linearno
python3 code/validate_accuracy.py --architecture linearno_conv
```

The durable cost controller waits for all accuracy receipts and for an empty physical GPU 2. It runs every backbone and both reconstruction implementations on that GPU, in randomized and interleaved order. It does not stop other GPU work:

```bash
taskset -c 16-23 python3 -u code/postprocess_workflow.py --gpu 2
```

The controller executes the 333 paired timing groups, 11 independent model/method memory processes, `code/finalize_latency.py` and `code/report_results.py`. Timing has 20 warmups plus 100 formal repeats, with explicit A/B/C boundaries and per-condition preserved raw repeats. The benchmark sets all three BLAS/OpenMP thread variables to 4 before importing numerical libraries. A failed profiling stage can resume from preserved direct timing groups.

The code patch and current source index are generated as `results/implementation.patch` and `results/implementation_sources.json`. Large checkpoints and head feature banks remain in this internal directory; the archive refers to them by path and hash. No command in this workflow pushes to GitHub or changes the manuscript or Feishu.

The latest user authorization permits the parent agent to publish a snapshot to an independent branch. Git publication is handled only by that parent; the Airfoil experiment commands above remain local and do not push to main.
