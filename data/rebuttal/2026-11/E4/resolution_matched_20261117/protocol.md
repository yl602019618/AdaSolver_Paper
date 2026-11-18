# Darcy resolution-matched training

The independent variable is the observation and training grid: 85×85, 141×141, or 211×211. Each resolution receives a separately trained Transolver3 and its own CA head. All choices below were fixed before the new training runs. Machine-readable settings are in `protocol.json`; the resource-only change for 141² is in `runtime/resource_amendment.json`.

## Data and model selection

The inputs are the original `coeff` and `sol` arrays in the two 421×421 Darcy MAT files. Direct strides 5, 3, and 2 select the three observation grids. The same physical cases are used at every resolution: smooth1 rows [0,900) for training, smooth1 [900,1000) for validation, and smooth2 [0,200) for evaluation. Global case IDs 1000–1199 refer to those smooth2 evaluation rows. Per-resolution scalar normalizers are fitted only on the 900 training cases. Array identities, shapes, and source node checks are saved in `data/`, `split_manifest.json`, and `config/source_node_checks.json`.

Existing 85² checkpoints do not match this contiguous split and selection rule; a new 85² internal control is therefore trained. The historical smooth2 population has been viewed in earlier work. This run excludes it from checkpoint, head, and strategy selection; it is not claimed to be a historically untouched holdout.

## Backbone training

Each model is initialized from scratch with training seed 0: hidden width 128, eight layers, eight heads, 64 slices, MLP ratio 2, ref 8, dropout 0. Training uses the complete native grid for 1000 epochs, effective batch 4, 225 updates per epoch, and 225000 updates in total. Two GPUs process local batches of two. AdamW uses lr 1e-3, weight decay 1e-5, gradient clipping 0.1, and OneCycleLR with pct_start 0.3. Computation is eager FP32 with TF32 disabled.

The existing 900/100 Darcy objective is preserved: the global batch sum of physical relative L2 plus 0.1 times the x- and y-derivative relative L2 terms. The derivative term retains the original zero predicted boundary collar and central differences with h=1/R. The full field output is not zeroed. Local summed losses are multiplied by the DDP world size so that gradient averaging reproduces the global summed objective. An independent check against the previous 85² loss reports exact loss and gradient equality in `results/loss_recipe_validation.json`.

The saved backbone is the epoch with the lowest mean physical native-grid relative L2 on the 100 validation cases; exact ties retain the earlier epoch. Full input, prediction shape, native point count, and split access are recorded by the preflight and production hooks. All methods within a resolution share this checkpoint and normalizer.

## CA heads and sampling

Each frozen backbone supplies its own block-0 features, normalized coarse predictions, and native-resolution teacher scores. Its two-block TransolverGradientHead is trained for 1000 epochs with seed 0, effective batch 4, AdamW/OneCycle settings matching the declared head configuration. Teacher targets are zscore(log1p(raw grad_lap)); the target normalization uses only the 900 training cases. The head is selected by validation mean per-case vector error norm. Mean Huber loss uses ordinary DDP averaging, unlike the summed backbone loss. Head banks, weights, and acceptance records are bound to the backbone and normalizer hashes.

PI keeps the predeclared shared beta=1, amp=0.25, ell=2/84 and the original E4 flattened-mask EDT. CPG and CA retain the original E4 grad_lap/index-Laplacian convention, 32 patches, kappa=2, beta=1, native exact FPS, and coarse_fraction=0.1. No strategy parameter is selected from this run's test errors. The coarse counts are 722, 1988, and 4452; all coarse points count toward the final budget. The known PI distance convention is preserved here so that its correction is not confounded with training resolution.

Each resolution evaluates Full once per case and Uniform, PI, CPG, and CA at budgets 0.375, 0.5, and 0.75 with sampling seeds 0, 1, and 2. Final point counts use `int(round(N*budget))`. There are 7400 physical prediction requests per resolution, 22200 overall. Retained indices and their hashes are saved.

## Reconstruction and statistics

The reconstruction rule remains the original E4 Delaunay linear interpolation plus nearest fallback. Reference coordinates use exactly the FP32 model coordinate values promoted to FP64. Full native output is the identity; Full at a lower observation resolution is reconstructed onto the common grid. Every prediction is scored in physical units on the same true 211² output grid. Native-grid relative L2 is also retained as a separate view, yielding 14800 metric rows per resolution and 44400 overall. No formal latency claim is made from single evaluation request durations.

For each resolution, checkpoint, grid scope, method, and sampling seed, first average relative L2 across all 200 cases at each budget. Apply trapezoidal integration over [0.375,0.5,0.75]. Divide that area by the same seed's matched Uniform area, then report the mean and sample SD (ddof=1) across the three ratios. Full is excluded from the integral. These are three sampling seeds under one trained model, not three training seeds.

## Regional evidence and presentation

Boundary, coefficient-interface, and smooth regions use the same physical definitions at every observation resolution. Boundary and interface widths are 0.02; the interface excludes the boundary. Boundary membership is evaluated directly at each native coordinate. Interface membership uses physical distance to coefficient-interface nodes identified on the true 211² coefficient field. Raw refinement scores, nominal or planned target allocation, and actual retained indicators remain distinct. Regional counts, within-region retention, and retention relative to the actual global budget are saved. Example physical cases 1000, 1099, and 1199 are fixed in advance. Shared display scales are used when comparing corresponding scores and retention maps.

The 2026-11-19 audit corrected the initial nearest-grid-label assignment, which had placed 301 native 141² nodes outside the prescribed physical boundary band into the boundary region. The original `region_records.csv` files remain as legacy records; the final regional analysis uses separately generated `physical_region_records.csv` and its acceptance receipt. This correction changes regional classification and summaries only. It does not modify the trained weights, selected points, PI distance algorithm, CPG derivatives, reconstruction, or full-field errors/rAUEC. The locked machine-readable protocol remains preserved, with the regional correction documented separately.

The requested rebuttal table contains the published original 85² row as a labeled historical reference plus new 141² and 211² results. Historical 85² used a different training/selection history, native85 evaluation, and PI beta=0.5. Only the newly matched 85²/141²/211² results support controlled resolution trends. The new 85² is not duplicated in the author-facing main table.

## Execution and records

Three parent controllers own idle GPU pairs through shared locks. They run backbone preflight, full training, own-head bank generation and training, head acceptance, and two disjoint evaluation shards. The pairs are H20 [1,2] for 85², [0,3] for 141², and [6,7] for 211². The original [4,5] allocation for 141² was replaced before its training began because it was occupied by other jobs. Scientific settings and the locked protocol hash are unchanged.

`python3 code/monitor.py` prints progress; its durable watch process updates `read.me` every 30 seconds. Run commands and completion gates are recorded in `runtime/*_workflow.json` and `config/post_backbone_commands.json`. Existing E4 results, upstream repositories, and manuscript edits are preserved. A complete result requires accepted training, own-head identities, all raw metric rows, independently recomputed rAUEC, and generated figures. Missing stages remain explicitly incomplete.
