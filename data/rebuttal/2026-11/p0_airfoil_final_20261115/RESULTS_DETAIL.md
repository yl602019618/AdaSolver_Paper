# Airfoil P0 results

Status: PARTIAL. Accepted complete training seeds: [].

The required study has two point models, training seeds0/1/2, their own heads,110 historical positive-lift test cases, and budgets37.5%,50%,75%. Full100% has no sampling seed. Accuracy and timing include Uniform, PI, CPG beta0/beta1 and CA. Each seed's complete timing matrix has414 groups and1,638 physical conditions over scopesA/B/C, with20 warmups and100 formal repeats. Seed0 reuses1,314 verified point-model conditions and adds only324 beta0 conditions; legacy conv-temp Full-only observations are excluded from the primary three-training-seed totals.

Observed counts: {"accuracy_requests": 10120, "formal_repeats": 131400, "full_reference_requests": 220, "heads": 2, "memory_conditions": 1314, "models": 2, "parity_pairs": 9900, "physical_conditions": 1314, "training_seeds": 1, "warmups": 26280}.

Accuracy aggregation first averages cases within a sampling seed, then sampling results within one checkpoint, then calculates mean and sampleSD across three independently trained models. Each rAUEC uses its own checkpoint/population/reconstruction matched Uniform denominator per sampling seed before aggregation. No training SD is filled when fewer than three models are available. Old and new reconstruction remain separate groups; the primary accuracy group is grid_qhull_sparse_apply_v1p1.

Latency preserves each checkpoint and physical GPU. ScopeA is CPU input to complete CPU float64 output; B has resident GPU model tensors and complete GPU float64 output with explicit CPU geometry work; C caches only the same geometry/ordered-point-set interpolation map. Per-condition timings and per-checkpoint/GPU pooled tables are retained; raw timings are never pooled across GPUs or training seeds. Between-checkpoint timing differences also include device and CPU-affinity differences and are not labeled pure training randomness.

The benchmark uses three fixed timing cases1001/1094/1198, three sampling seeds for adaptive methods, FP32 networks and complete float64 fields. Memory is measured in separate model/method processes. First-plus-steady cache costs are derived estimates, not contiguous request wall times. backbone_latency_summary.csv retains genuine pooled formal wall-time median/p95 within each checkpoint/training-seed/GPU. latency_summary.csv is the condition table. latency_repeats.csv.gz is a lossless gzip-compressed CSV containing the requested latency_repeats.csv scalar observations.

Pending: ["training_seed0: 1314/1638 latency+memory conditions", "training_seed1: no accepted data-file index", "training_seed2: no accepted data-file index"].

All numeric tables contain observed results only. Verification requires6 model/head pairs,30,360 accuracy requests,4,914 timing/memory conditions and491,400 formal repetitions before final status. Complete CFD truth, weights, feature caches and long logs are not published.

CSV-only verification: python code/validate_results.py --data-dir . [--require-complete]
Figure reproduction: python code/plot_final.py --data-dir . --output-dir figures
