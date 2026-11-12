# E3: Matched Transolver3 and LinearNO

Each architecture uses a frozen, current-environment 400-epoch DDP2 seed-0 checkpoint, the same 710/79/100 train/validation/test partition and matched training budget. All 100 test cases and three sampling seeds are included. Accuracy, all 276 unique timing conditions, 27,600 formal repetitions, alias views and 276 isolated-memory conditions passed independent checks.

Controlled latency measures CPU input to complete CPU field, using eager FP32 networks; the output container is float64 and force integration is outside the timing boundary. Full median latency is 17.6848 ms for the project's matrix-optimized T3 and 44.6465 ms for the fixed official LinearNO implementation. Every measured CPG pooled median exceeds matched Full. LinearNO Full Field error is lower, but this implementation comparison does not establish architecture-intrinsic speed ordering. Its Field beta=1 rAUEC (0.531376) is slightly above beta=0 (0.527244).

Official LinearNO source: https://github.com/HiPRL/LinearNO ; paper: https://arxiv.org/abs/2511.06294 . Third-party code and pretrained weights are not redistributed here.

## Files

- [backbone_summary.csv](backbone_summary.csv)
- [backbone_rauec.csv](backbone_rauec.csv)
- [backbone_accuracy_latency.csv](backbone_accuracy_latency.csv)
- [backbone_latency_summary.csv](backbone_latency_summary.csv)
- [backbone_training_cost.csv](backbone_training_cost.csv)
- [checkpoint_registry.csv](checkpoint_registry.csv)
- [isolated_memory.csv](isolated_memory.csv)
- [metrics_records.csv](metrics_records.csv)
- [controlled_latency_repeats.csv](controlled_latency_repeats.csv)
