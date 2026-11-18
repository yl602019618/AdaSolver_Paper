# Reproduce the matched Darcy experiment

Run commands from this directory. The original inputs are the 421² `coeff` and `sol` MAT arrays identified in `data/prepared.json`. Dependency versions and hardware are in `config/environment.json`. The task-local `code/vendor/adasolver/` snapshot is the model and sampler implementation; do not substitute a different installed package.

## Data and independent input checks

```bash
python3 code/prepare_data.py
python3 code/verify_design.py --data-only
```

Preparation is idempotent for the locked protocol. The verifier compares every saved train, validation, and test element with its original 421² source node. It independently checks training-only normalization statistics. Existing `data/` files must not be replaced while a training job is running.

## Complete per-resolution workflows

Use one command per terminal. The controllers acquire shared idle-GPU locks and will wait if the pair is occupied. The current run used these GPU pairs:

```bash
python3 code/run_resolution.py --resolution 85 --gpus 1,2
python3 code/run_resolution.py --resolution 141 --gpus 0,3
python3 code/run_resolution.py --resolution 211 --gpus 6,7
```

Do not start a second controller for a resolution already running. `runtime/r*_workflow.json` records its PID, stage, command, and heartbeat. A resource override changes only GPU assignment. It retains the same two-rank effective-batch-4 experiment. The parent controller preserves checkpoints, uses explicit resume when needed, and executes the head/evaluation contracts in `config/post_backbone_commands.json`.

The underlying backbone command is:

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 code/train_ddp.py --resolution 211 --microbatch-size 2
```

The parent sets `CUDA_VISIBLE_DEVICES`, CPU thread counts, and the native FPS cache. Direct child commands should be run only under an owned idle GPU pair. A two-step engineering preflight uses `preflight_training.py` with the same distributed arguments. It does not replace any production epoch.

After the corresponding backbone completes, the controller reloads its selected checkpoint and re-evaluates the 100 validation cases with `verify_training.py --resolutions 211 --reload-validation --device cuda:0`. This checks the saved weight file against the validation-selected epoch. The three head commands then follow:

```bash
python3 code/prepare_head.py --resolution 211 --device cuda:0 --resume
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 code/train_head.py --resolution 211 --resume
python3 code/verify_head.py --resolution 211 --device cuda:0
```

The parent then runs `python3 code/evaluate_pair.py --resolution 211`. The wrapper runs two disjoint case shards on its two visible GPUs and independently checks the merged matrix. Replace 211 with 85 or 141 for the corresponding native model, head, and data. Neither the head nor the evaluator silently accepts unfinished backbone training.

## Progress and final calculations

```bash
python3 code/monitor.py
python3 code/finalize_results.py --watch
```

The current run already has a durable monitor and finalization watcher; their PIDs are recorded in `runtime/monitor.json` and `runtime/finalization_launch.json`. `read.me` and `runtime/status.json` are updated every 30 seconds. The finalizer waits for all three accepted accuracy matrices, then runs:

```bash
python3 code/verify_training.py
python3 code/verify_design.py
python3 code/recompute_physical_regions.py
python3 code/verify_physical_regions.py
python3 code/aggregate.py
python3 code/verify_statistics.py
python3 code/plot_results.py
```

The statistics verifier reads the raw per-case CSVs and independently implements scalar means, trapezoidal areas, matched Uniform ratios, and sample SD. The regional postprocessor uses the prescribed 0.02 physical band directly at native coordinates and writes `physical_region_records.csv`; the original nearest-grid-label `region_records.csv` remains as a legacy record. Its independent verifier checks labels against direct physical distances and checks the complete count/retention matrix. Full-field metrics and selected indices are unchanged. The plots are generated from saved CSVs and actual selected-point arrays. `results/ready_for_review.json` marks numerical completion and generated figures; visual review precedes the final acceptance record.

The raw accuracy matrix has 7400 physical requests per resolution: 200 Full requests plus 200×4 methods×3 budgets×3 sampling seeds. Common211 and native are two metric views of each request, for 44400 metric rows overall. Full has no artificial sampling seed and is excluded from rAUEC.

## Result paths

- `checkpoints/r{85,141,211}/`: each backbone, its own `head/`, source snapshots, configurations, and training curves.
- `head_banks/r{85,141,211}/`: frozen features, teacher predictions, targets, coarse indices, and identities.
- `records/r{85,141,211}/`: per-case JSON/CSV records, retained points, compensation weights, and physical predictions.
- `evaluation/`: reconstruction and interface checks, example arrays, and shard receipts.
- `results/`: common/native errors, per-seed rAUEC, sampling mean/SD, regional statistics, tables, and figures.
- `REPORT.md`, `CLAIM_CARD.md`, `read.me`: interpretation, claim scope, and current completion status.

The author-facing main table uses original paper85 as a clearly labeled historical reference plus new141/new211. The controlled resolution analysis uses the newly trained internal85/new141/new211 cohort. The protocol remains explicit about the historical population, PI setting, and evaluation-grid differences.
