# P0 Airfoil final evidence protocol

<!-- original-cdcl-correction-20261117 -->
**Cd/Cl scope correction, 2026-11-17.** The Cd/Cl rows and panels in this P0 package are **field-policy force diagnostics**. They cannot be compared with the original Table 1 Cd/Cl metric-specific policies. The active Transolver3 three-training-seed Cd/Cl result is the [corrected original-policy package](../p0_airfoil_original_cdcl_20261117/README.md). Current Cd/Cl summaries and the reviewer response cite that package. The P0 raw numerical files, field accuracy, reconstruction parity and timing remain preserved under their original policies. The correction does not establish an original-policy Cd/Cl result for LinearNO.
<!-- /original-cdcl-correction-20261117 -->

The paper evidence baseline remains91c1fa7cc81b8c1cb61a43618296c1c892d818c0. P0-A is recorded in reuse_audit_final.json before new GPU work. Seed0 training, heads, complete accuracy and the existing four-adaptive-method timing matrix are reused; the active Transolver3 training seed1 is adopted without restart.

Airfoil uses the locked900 training/100 validation split,900-only normalizer, published110 positive-lift test IDs,221×51 grid,1000 backbone epochs and113000 updates. Only training and head seeds vary across0/1/2. Every CA head consumes its own frozen backbone features. Checkpoint and head selection use validation only. The test population has historical exposure and is not relabelled an untouched holdout.

For each frozen model, evaluate Full100% once per case; Uniform, PI, CPG beta0, CPG beta1 and CA at0.375/0.50/0.75 with sampling seeds0/1/2. CPG beta pairs share coarse prediction and selected indices. Old/new reconstruction IDs are grid_scipy_linear_nearest_native_v1 and grid_qhull_sparse_apply_v1p1. Model-specific parity must satisfy the locked1e-6+1e-5*abs(old) elementwise bound before accepting the main new-reconstruction accuracy tables.

Case means and sampling-seed quantities are aggregated inside each training model first. Only then are mean and sampleSD(ddof1) computed across the three independent training models. Full is excluded from rAUEC; each sampling seed uses the same-model/checkpoint/population/reconstruction Uniform area over the three budgets. Training SD, sampling SD and timing-repeat SD are separate.

A is CPU model-ready input to complete float64 CPU field. B is resident GPU input to complete float64 GPU field, including any actual CPU work/transfers. C reuses only the interpolation map for identical geometry and ordered selected indices; acquisition, model prediction and necessary dynamic work remain part of the request. First preparation and map application/reuse are separate. Direct synchronized wall time is authoritative; stage profiles are not summed into a substitute total.

The P0 point-only matrix has414 paired groups and1638 physical conditions per training seed:18 Full conditions plus1620 adaptive conditions. Each uses20 warmups and100 formal repeats. Seed0 already has1314 verified point conditions; only81 beta0 groups/324 conditions and corresponding isolated-memory conditions are added. The9 existing conv Full conditions remain a separate supplement. Across all three point-model training seeds, the target is4914 conditions and491400 formal repetitions;3600 conditions/360000 repetitions are new work.

Training resources: preserve Transolver3 seed1 onGPU0; launch LinearNO seed1 onGPU1, Transolver3 seed2 onGPU6 and LinearNO seed2 onGPU7. Timing usesGPU2 for seed0 supplement and seed1;GPU3 for seed2. Every training-seed-specific comparison uses one physical H20 for both backbones, all methods and both reconstructions. GPU identity, CPU affinity, dtype, cache keys and memory boundaries are recorded. Latency rows from different physical GPUs are not silently pooled into one estimate. The original stage runners and the P0 adapter enforce exclusive admission for their selected GPU.

The earlier serial dispatch process is replaced for P0 scheduling only; its running training worker and all existing results remain intact. Other unfinished task definitions are preserved. No failed experiment is silently relaunched.

Default deadline snapshot:16 November2026 at09:00 Asia/Shanghai (01:00UTC), pending any user-supplied replacement. At the deadline, publish only verified available evidence and actual missing counts, leaving running work alive. When all new seed and timing gates pass, automatic final aggregation and the target-branch publication must succeed before the P0 terminal task is complete. No training-variance or new-checkpoint acceleration claim is made before the corresponding data are accepted.

The terminal controller task invokes finish_workflow.py --mode final --execute. It returns success only after complete numerical/visual acceptance and verified remote publication. The deadline event invokes --mode snapshot --execute, keeps all GPU work alive and cannot overwrite an already published final result. Stage events update status only.

Cross-task P0-A inventory: reuse_audit_final_addendum.json revalidates the existing matched Darcy seed0 records and reconstruction summary without restarting that completed work. Darcy training seeds1/2 remain a separate pending cohort, outside the prioritized Airfoil execution.
