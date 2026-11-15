# Full-field reconstruction: current verified measurements

This package tests whether replacing the original reconstruction with the rule-equivalent sparse application changes complete-field accuracy, request latency, memory, and fixed-map reuse cost. It contains newly completed Car and Airfoil measurements, independently aggregated from preserved raw records. No training, GPU execution, checkpoint selection, or old experiment was repeated for this export.

| Dataset/model | Accuracy test | Sampling and budget | Formal timing | Memory |
|---|---|---|---|---|
| ShapeNet-Car Transolver3 seed0 |100 historical cases; Full, Uniform, PI, CPG beta0/beta1, CA; own head |sampling seeds0/1/2;37.5%,50%,75%; Full100% |A/B/C;3 fixed cases;20 warmups and100 formal repeats per physical condition |isolated model/method process; returned output and cached-map allocations included |
| ShapeNet-Car LinearNO seed0 |100 historical cases; Full, Uniform, CPG beta0/beta1 |same three sampling seeds and budgets |same boundary and repeats |same allocator scope |
| Airfoil221×51 Transolver3 seed0 |110 published positive-lift cases; Full, Uniform, PI, CPG beta0/beta1, CA; own head |sampling seeds0/1/2;37.5%,50%,75%; Full100% |A/B/C;cases1001/1094/1198;20 warmups and100 formal repeats |isolated model/method process |
| Airfoil official LinearNO point-temp seed0 |same110 cases and five method families; own head |same three sampling seeds and budgets |same boundary and repeats |same allocator scope |
| Airfoil official LinearNO conv-temp seed0 |Full-only on110 cases; distinct official model class |Full100%, no sampling seed |one Full series per case and scope |Full-only isolated process |

Car contributes7,400 old/new accuracy pairs,1,800 matched beta pairs,990 timing conditions,99,000 formal repetitions and990 memory conditions. Airfoil contributes10,230 accuracy requests including the Full-only model,333 paired timing groups,1,323 timing conditions,132,300 formal repetitions and1,323 memory conditions. Full is measured once per model/case/scope and its old/new views are aliases. Full is not inserted into the rAUEC integral.

The main A boundary is CPU input to a complete CPU float64 field, with batch1 and eager FP32 networks. Every changing-point-set request builds its interpolation map. B starts from resident GPU model tensors and ends with a complete GPU float64 field; the explicitly timed CPU geometry/sampling work remains included. C has the CPU-to-CPU boundary with a cache for the same geometry and ordered selected set; only interpolation maps are reused, while coarse acquisition, score, selection and model prediction are recomputed. C old is equivalent cached CPU sparse application; C new is cached GPU application. These cache scenes are not compared to A Full as if they were the same request.

All old/new accuracy comparisons bind the same checkpoint, prediction, selected indices, normalizer and metric population. The timing output checks are independently matched to the corresponding accuracy records. Car's original surface-aware k4 IDW and Airfoil's original linear interpolation plus nearest fallback remain distinct reconstruction protocols. Error tables retain the original physical metrics and both reconstruction IDs.

Each rAUEC is computed separately for sampling seed0/1/2 by integrating the three budget errors and dividing by that seed's matched Uniform area. Reported uncertainty is sample SD across sampling seeds; these data contain one independent training model per listed backbone, not three. The broader three-training-seed requirement is not declared complete by these measurements. Car LinearNO PI/CA remain unavailable in the preserved Car reconstruction protocol; no Transolver head is attached to LinearNO.

The following50%-budget examples use pooled A-boundary request medians in milliseconds:

| Dataset/model/method | Original | New | Interpretation |
|---|---:|---:|---|
| Car / Transolver3 / Uniform |61.5729 |28.2630 |new complete request faster |
| Car / Transolver3 / CPG beta1 |270.6697 |237.0018 |new complete request faster; acquisition/selection remains substantial |
| Car / LinearNO / Uniform |75.4260 |42.0195 |new complete request faster |
| Car / LinearNO / CPG beta1 |284.4168 |250.3719 |new complete request faster |
| Airfoil / Transolver3 / Uniform |67.6829 |68.7727 |new complete request slower |
| Airfoil / Transolver3 / CPG beta1 |94.6335 |95.7804 |new complete request slower |
| Airfoil / point-temp / Uniform |69.1581 |70.1927 |new complete request slower |
| Airfoil / point-temp / CPG beta1 |94.0290 |95.2957 |new complete request slower |

Across all72 Airfoil old/new grouped comparisons over A/B/C,48 new variants are slower. The old/new median ratio ranges from0.9719565 to1.0215509. All measured rows are retained. The sparse application does not imply an end-to-end speedup when the full request still pays geometry, QHull/search, sampling and transfer costs.

Cache tables report the measured first request, steady-state request median, retained CPU/GPU map bytes and a derived first-plus-reuse estimate. Airfoil additionally preserves assembled sums of the first request and subsequent post-warmup observations. Neither form is a contiguous n-request wall-time measurement. Reconstruction stage medians/p95 come from separate profiling passes, never from summing stages to replace directly observed total request time. PyTorch allocator peaks do not certify CPU allocations or all transient external GPU allocations; post-request NVML evidence is bounded separately.

The original Airfoil finalizer failed after all measurements completed because dict(**meta, reuse_assumption=...) repeated a key already present in metadata. This package recovers reporting in a new directory through explicit dictionary merging, checks all raw counts/repeats/source identities/memory joins, and preserves the original failure log and status. No original file was changed.

Main files:

- car_reconstruction_conditions.csv and airfoil_reconstruction_conditions.csv: every measured case/seed/budget/implementation/scope, direct median/p95, profile cost, memory, cache bytes, checkpoint and source identity.
- car_latency_grouped.csv and airfoil_latency_grouped.csv: pooled direct-request summaries over the fixed case/seed matrix.
- *_accuracy_budget.csv, *_sampling_seed_errors.csv and *_rauec.csv: complete accuracy and correctly separated sampling variability.
- *_reconstruction_parity.csv: same-prediction old/new numeric agreement.
- car_reuse_costs.csv and airfoil_cache_reuse_costs.csv: first construction and modeled reuse costs.
- *_raw_timing_repeats.csv.gz: compact scalar timing observations; no CFD fields or model weights.
- airfoil_stage_profiles.csv.gz: separately instrumented scalar stage observations. The original uncompressed82MB export is kept locally and excluded from publication.
- source_index.csv and audit.json: original record hashes and scoped acceptance evidence.
- export_validation.json: a second, CSV-only recomputation of condition and pooled timings, rAUEC and reuse estimates.

Read-only source export:

    python audit_export.py --workspace /path/to/07_current --output /path/to/new/export

Independent verification of the exported CSVs:

    python verify_export.py --data-dir . --output-dir .

The verifier prefers the gzip scalar-repeat files and also accepts uncompressed CSVs. The three compressed files contain LF CSV bytes; compression_manifest.json records both uncompressed and compressed hashes.

Regenerate all seven result figures/tables from the plain summary CSVs:

    python plot_results.py --data-dir . --output-dir figures

The figure families cover per-backbone rAUEC with sampling SD, complete error against budget, same-checkpoint error against direct scopeA latency, old/new full-request cost, scopeC cache reuse curves, a field-rAUEC table and a Full reference table including the conv-temp model. Each has PDF, editable SVG,300dpi PNG, source CSV and a caption. CSV-only numerical checks require NumPy2 or later; rendering additionally uses Matplotlib and Pillow. Neither verification nor plotting loads a model or uses a GPU.

The source datasets retain historical test exposure. The new Airfoil models use900 training cases and100 validation cases with validation-based model/head selection, but evaluation on the existing110-case population is not relabeled a never-touched holdout.

Published reproduction scripts are under `reproducibility/rebuttal/2026-11/current_results_20261115/reconstruction_airfoil`. Commands using script filenames refer to that code directory; the data directory is this package. Source manifests and original audit hashes are provenance snapshots; the current branch-wide byte hashes are in ../release_manifest.json.
