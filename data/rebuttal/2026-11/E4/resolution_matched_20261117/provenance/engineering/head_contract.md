# CA head implementation and acceptance contract

The resolution-specific head implementation is ready for the post-backbone pipeline. This is an implementation/preflight record, not a completed head-training or test-accuracy result. The exact source hashes and CPU acceptance values are recorded in `head_preflight.json`.

For each resolution `N` in 85, 141 and 211, run these commands with the parent pipeline's assigned GPU pair and working directory at the experiment root:

```bash
python3 code/prepare_head.py --resolution N --device cuda:0 --resume
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 code/train_head.py --resolution N --resume
python3 code/verify_head.py --resolution N --device cuda:0
```

The parent owns GPU leases. These commands only take their own bank/training file locks and do not acquire or change GPU leases. They never select a checkpoint or strategy with test data.

The bank generator requires the corresponding completed 1,000-epoch, 225,000-update backbone and its exact normalizer. It reads smooth1 training rows 0:900 and validation rows 900:1000 from their prepared native-resolution coefficient files. It does not read physical solution arrays or smooth2 test arrays. It normalizes the coefficient using the backbone's training-only normalizer, obtains the frozen first-block features and normalized coarse prediction, and computes the original raw grad-plus-index-Laplacian teacher at the same native resolution. The source data, native coordinates, normalizer, backbone file and actual loaded tensors are bound to the bank.

| Native resolution | Full source points | Coarse points | Frozen feature shape per case | Teacher grid |
|---|---:|---:|---|---|
| 85 × 85 | 7,225 | 722 | 722 × 128 | 85 × 85 |
| 141 × 141 | 19,881 | 1,988 | 1,988 × 128 | 141 × 141 |
| 211 × 211 | 44,521 | 4,452 | 4,452 × 128 | 211 × 211 |

The teacher retains the existing physical-coordinate squared-difference gradient and the existing index-space Laplacian. The target is `zscore(log1p(raw_teacher))`, with its mean/std fitted on only the first 900 bank rows. Head output is decoded as `expm1(clip(z * std + mean, 0, None))`; it is not squared again. Each physical case gets one fixed coarse mask from `SeedSequence([0, case_id, 20261114])`.

The native two-block `TransolverGradientHead` has 236,049 parameters. Every resolution starts its own seed-0 head from scratch and trains for 1,000 epochs, 225 optimizer updates per epoch, effective batch 4. With two DDP ranks, each rank processes local batch 2. The loss is the local **mean** Huber loss; DDP's gradient average supplies the intended global mean. If microbatch accumulation is used, each microbatch is scaled by `micro_batch / local_batch`. The backbone's global-sum-loss world-size multiplier must not be applied to the head. Validation checkpoint selection uses the mean per-case vector error norm on the 100 held-out bank rows.

The CPU preflight used four real training coefficient cases at each complete native grid and the actual 8-block backbone/2-block head classes. Frozen feature extraction matched the real first-block hook exactly. The native teacher and normalized CPG score differences were zero at all three resolutions. Two optimizer steps per resolution produced finite predictions/losses/gradients and left the backbone unchanged. The global-batch-4 gradient versus averaged local-batch-2 gradients agreed to relative norm 1.72e-7. These temporary initialized models were discarded and are not experiment checkpoints.

Completion is reported by `checkpoints/rN/head/manifest.json` with `execution_status=completed`, 1,000 epochs, 225,000 logical optimizer updates, `checkpoint_sha256` and its `head_sha256` alias. The final verifier additionally checks the complete history and validation-min epoch, optimizer/scheduler updates, every coarse mask, every standardized target, source identities, native teacher spot checks and fresh feature/prediction reproduction. It writes `results/head_rN_validation.json`; evaluation's `load_head` requires this receipt by default.

Training is resumable in 10-epoch intervals. A recovery preserves superseded history and best weights, restores the generator/optimizer/scheduler and the best checkpoint from the saved epoch, and records additional executed updates/time. An existing initial-only head file is preserved and verified against seed 0 when `--resume` is used. Source snapshots prevent numerical implementations from silently changing across a resume.

The earlier 85² same-architecture single-GPU 1,000-epoch head took 1,440.66 seconds (24.0 minutes), as recorded in `linearno_followup_20261114/standard_benchmarks/darcy/checkpoints/transolver3/head/manifest.json` relative to the workspace root. This is a historical planning reference. The current DDP2 and 141²/211² head durations have not yet been measured; ETA should be based on the first completed head epochs, and final training costs on the actual manifests.
