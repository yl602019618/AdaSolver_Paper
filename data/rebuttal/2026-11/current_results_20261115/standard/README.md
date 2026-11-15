# Standard benchmarks: three independent training seeds

This package independently verifies the completed Transolver3 study on Elasticity, Pipe and Darcy: **9 backbones, 9 corresponding CA heads and 66,600 full-field observations**. It reuses the completed experiments and their original records; no model inference, training or GPU timing was repeated by this audit.

| Dataset | Training / validation / test | Reference points | Backbone epochs / updates | Head epochs / device |
|---|---|---:|---:|---|
| Elasticity | 1000 / 200 / 200 | 972 | 500 / 500000 | 200 / CPU FP32 |
| Pipe | 1000 / 200 / 200 | 16641 | 1000 / 250000 | 1000 / GPU FP32 |
| Darcy, historical cohort | 1000 / 200 / 200 | 7225 | 1000 / 250000 | 1000 / GPU FP32 |

Each dataset has training seeds 0, 1 and 2. Every trained model uses sampling seeds 0, 1 and 2 at budgets 0.375, 0.50 and 0.75 for Uniform, PI, CPG and CA. Full uses the same checkpoint at 100% once per test case, with no sampling seed.

## Verified rAUEC

Values are mean ± sample SD across the three independent training runs. For each training run, the method/Uniform trapezoidal area ratio is computed separately for each sampling seed, then the three ratios are averaged. Full is excluded from the integration.

| Dataset | Uniform | PI | CPG | CA |
|---|---:|---:|---:|---:|
| Elasticity | 1.000000 ± 0.000000 | 0.726406 ± 0.002168 | 0.467020 ± 0.001175 | 0.560134 ± 0.010647 |
| Pipe | 1.000000 ± 0.000000 | 0.855711 ± 0.012698 | 0.654441 ± 0.010113 | 0.692265 ± 0.009394 |
| Darcy | 1.000000 ± 0.000000 | 0.995973 ± 0.002841 | 0.639290 ± 0.015871 | 0.639412 ± 0.016405 |

[Training-seed figure](../../../../../figures/rebuttal/2026-11/current_results_20261115/standard/standard_training_seed_rauec.pdf) · [SVG](../../../../../figures/rebuttal/2026-11/current_results_20261115/standard/standard_training_seed_rauec.svg) · [300dpi PNG](../../../../../figures/rebuttal/2026-11/current_results_20261115/standard/standard_training_seed_rauec.png) · [Figure source CSV](../../../../../figures/rebuttal/2026-11/current_results_20261115/standard/standard_training_seed_rauec_source.csv)

## What was tested

| Check | Verified scope |
|---|---|
| Completed training | Full epoch histories, validation-minimum selection, every persisted optimizer step counter, scheduler steps, strict state loading and exact last/resume state equality for 9 backbones and 9 heads. |
| Model identity | Actual best/last/head/normalizer SHA256 values, matching backbone/head identities, frozen split IDs and complete corresponding validation selection. |
| Full accuracy matrix | 7400 observations per trained model; exactly 200 cases × (1 Full + 4 methods × 3 budgets × 3 sampling seeds); unique conditions and finite errors. |
| Raw numeric identity | Every relative-L2 value recomputed from its recorded numerator/denominator; public compressed observations checked row-for-row against the 90 original JSONL files. |
| Selected points | All 81 sampling archives checked for shapes, uniqueness and point bounds; CPG/CA coarse prefixes reproduce the declared RNG construction; Uniform/PI selections match across training models. |
| Statistical aggregation | 39 absolute-error rows,117 sampling-variability rows,12 three-training-seed rAUEC rows,108 per-training/per-sampling area ratios, and54 paired-difference rows independently recomputed. |
| Numerical agreement | Maximum absolute difference from existing tables: `2.22e-16`. |
| Timing scope | Training wall-time bindings were checked. Formal inference latency and GPU busy time are outside this audit. |

## Protocol and recovery notes

Elasticity uses training IDs 0–999, validation 1000–1199 and test 1600–1799. Its seed 0 head initially failed at a startup GPU-file lock. The failure log is preserved. Recovery is verified by its complete 200-epoch CPU history, best/last/resume weights, optimizer counters, completed runtime state and 7400 evaluated observations. All three Elasticity heads consistently used CPU FP32. The original failure log does not contain the later successful stdout or exact recovery launch command; the retained successful runtime/history supplies that completion evidence.

Pipe trains on IDs 0–999, validates on 1200–1399 and tests on 1000–1199. Its frozen sampling policies were historically calibrated on the first 50 cases of that test population.

Darcy trains on smooth1 rows 0–999, validates on smooth2 rows 200–399 and tests on smooth2 rows 0–199. This historical 1000/200 cohort uses normalized-output training loss and 250000 updates. It is an alternative protocol to the separate 900/100 matched Transolver3/LinearNO cohort, which uses a different split, normalizer and physical-plus-derivative loss. The seed values must not be pooled across these cohorts. Darcy policy development was historically test-informed; the current checkpoint selection used the separate validation cohort.

## Source release and files

The source snapshot is `3ef2190d2393ff4a6c2af9d2675a9385651e2d6a`, based on the fixed paper commit `91c1fa7cc81b8c1cb61a43618296c1c892d818c0`. All 53 source-commit objects were inspected; all published CSV values match the audited original tables. The source release manifest has 6 stale CSV hash/byte entries caused by CRLF-to-LF conversion. This source metadata conflict is preserved in `audit.json`; the destination release must hash its own final bytes. The source root README is not treated as the destination branch README.

- `audit.json`: independent checks, exact scope, source-commit verification and preserved failure/recovery evidence.
- `checkpoint_registry.csv`: 9 complete model/head pairs, classes, epochs, update counts, validation values, training wall times and hashes.
- `training_seed_summary.csv`, `training_seed_rauec.csv`, `sampling_variability.csv`, `paired_training_differences.csv`: numerical summaries.
- `rauec_per_training_sampling_seed.csv`: method areas and matched Uniform denominators, keyed by checkpoint and population.
- `raw_records_index.csv`, `source_hashes.json`: original external records and immutable Git objects, including byte sizes and SHA256; no weights or long-log payloads.

Reproduce the source audit with the external original experiment workspace:

```bash
CUDA_VISIBLE_DEVICES='' python3 audit_standard_results.py --workspace "$ADASOLVER_WORKSPACE" --release-commit 3ef2190d2393ff4a6c2af9d2675a9385651e2d6a --output fresh_audit
python3 plot_standard_training_seeds.py --data training_seed_rauec.csv --output-dir figures
```

The numerical source-commit data are already available in `data/rebuttal/2026-11/E5_standard/`, including the public compressed per-case observations. The prepared commit is provenance; current target-branch push status is owned by the parent publication record.

Published reproduction scripts are under `reproducibility/rebuttal/2026-11/current_results_20261115/standard`. Commands using script filenames refer to that code directory; the data directory is this package. Source manifests and original audit hashes are provenance snapshots; the current branch-wide byte hashes are in ../release_manifest.json.
