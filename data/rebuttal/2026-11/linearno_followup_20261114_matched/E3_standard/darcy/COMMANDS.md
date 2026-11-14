# Reproducible local commands

Run from this `darcy/` directory. Existing controllers and outputs must be inspected before launching a duplicate run. The live process records are in `runtime/launches.json` and `runtime/postprocess_launch.json`; no second set should be started while those processes remain active.

The matched training/evaluation queues were launched as:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 taskset -c 0-7 python3 -u code/workflow.py --key transolver3 --gpu 0
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 taskset -c 8-15 python3 -u code/workflow.py --key linearno --gpu 1
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 taskset -c 40-47 python3 -u code/workflow.py --key linearno_conv --gpu 5
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 taskset -c 0-7 python3 -u code/postprocess_workflow.py --gpu 0
```

The point-model stage sequence is `train_backbone.py`, `evaluate.py --phase baseline`, `prepare_head.py`, `train_head.py`, then `evaluate.py --phase ca`. `--backbone Transolver3` or `--backbone LinearNO` selects the registered native classes. The convolution model adds `--variant conv` to training and `--conv-only` to its Full evaluation.

Global collectors should use `records/{model}/baseline_completion.json` or `complete_completion.json` and the hash-bound `metrics_csv` path in that receipt. `metrics_records.csv` remains the live compatibility export and can be extended while CA runs. Phase snapshots are published once and never overwritten with conflicting content.

After completed model evaluation, the postprocess queue runs:

```sh
python3 code/finalize_accuracy.py
CUDA_VISIBLE_DEVICES=0 python3 code/benchmark.py --phase paired --gpu 0
CUDA_VISIBLE_DEVICES=0 python3 code/benchmark.py --phase memory --gpu 0 --backbone Transolver3 --method full
CUDA_VISIBLE_DEVICES=0 python3 code/benchmark.py --phase memory --gpu 0 --backbone LinearNO --method full
CUDA_VISIBLE_DEVICES=0 python3 code/benchmark.py --phase memory --gpu 0 --backbone LinearNO --conv-only --method full
python3 code/finalize_latency.py
python3 code/report_results.py
```

The two point-model memory commands are repeated in separate processes for `uniform`, `pi`, `cpg` and `ca`, giving 11 memory processes in total. GPU admission and occupancy monitoring apply to every timing process. `finalize_latency.py --allow-partial` can inspect actual completed groups but marks incomplete matrices as partial.

Interface, objective, complete-request and timing-persistence checks are recorded in `results/`. The complete-request check uses initialized full model classes and is explicitly not a trained-accuracy experiment. Production accuracy and timing cannot run without completed checkpoint and reconstruction gates.

Backbone and head trainers accept an explicit `--resume` after a documented interruption. The queue does not silently overwrite an existing partial run. Accuracy or memory partial outputs likewise require preserving their failed attempt before an explicit recovery; completed direct timing groups are retained independently of profiling failures.

Parent-owned cross-dataset aggregation and publication figures consume the CSV files in `results/`. Large checkpoints, prediction arrays and logs remain in this internal package and are indexed by paths and hashes rather than copied into the paper repository.
