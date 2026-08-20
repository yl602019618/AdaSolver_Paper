# AdaTransolver draft notes

## One-sentence argument

In neural-operator inference under a finite spatial budget, AdaTransolver combines variable-cardinality training, measure-aware Physics-Attention, and adaptive allocation so that a frozen model can spend more points in high-value regions without changing the physical measure represented by point aggregation; current evidence supports equal-final-point accuracy gains, not yet a universal equal-time scaling claim.

## Terminology ledger

| Canonical term | First-use definition | Variants in source | Decision |
|---|---|---|---|
| AdaTransolver | Measure-aware spatial test-time scaling framework | Ada-Transolver, AdaSolver, AdNO | Use `AdaTransolver` for the paper-level framework. |
| variable-cardinality training | Uniform subset training with a randomly varying point count | varpts, Ada training, variable-N | Define once; use `variable-$N$` only in equations and compact labels. |
| measure-aware Physics-Attention | Point-to-slice aggregation weighted by represented physical mass | SNIS compensation, density correction, quadrature Physics-Attention | Use the architecture-level term in prose and SNIS for the estimator. |
| spatial test-time scaling | Reallocating or increasing spatial samples with frozen model parameters | test-time adaptive refinement, inference-time densification | Use this as the central problem name. |
| coarse-field allocation | Two-pass policy driven by a coarse predicted field | scheme 1, gradient policy, AdNO | Use `coarse-field allocation`; name the gradient score when needed. |
| learned acquisition head | Lightweight head predicting the refinement signal | scheme 3, grad head, multihead coarse | Avoid `multihead` unless the exact head architecture is being discussed. |
| relative area under the error--budget curve (rAUEC) | Method AUEC divided by uniform-sampling AUEC | RAUEC, rAUEC | Use `rAUEC`; lower is better and uniform is 1.0. |
| density-ratio cap | Maximum-to-minimum patch-density bound | kappa, aggressive factor | Define as $\kappa$ once. |

## Section outline

1. Introduction: distinguish irregular-input compatibility, measure consistency, and adaptive allocation.
2. Related work: irregular-geometry operators; discretization consistency; adaptive meshes and test-time refinement.
3. Method: problem, variable-cardinality training, measure-aware Physics-Attention, policy-decoupled allocation, metrics.
4. Experiments: five PDE benchmarks, NACA analysis, DrivAerML, diagnostic failures.
5. Discussion and conclusion: interpretation, finite-sample correction trade-off, and claim boundaries.

## Claim-evidence map

| Claim | Evidence currently in repository | Status |
|---|---|---|
| Coarse-field allocation improves equal-final-point rAUEC on five PDE benchmarks. | `README.md` and `02_inference/method2_coarse_grad/pde_benchmarks/report/OVERALL_REPORT.md`; rAUEC 0.574--0.937. | Supported, but consolidate seed files and confidence intervals. |
| NACA reaches full-grid accuracy with about 75% of final points. | `FINAL_adaptive_discretization.md`; adaptive error 0.0062 versus full-grid 0.0061. | Supported within reported rounding. |
| Measure correction removes point-replication drift. | `02_inference/method1_physical_prior/airfoil/RESULTS.md`; 0.02 to 4.7e-7. | Supported as an implementation audit, not a generalization theorem. |
| The learned acquisition head reduces acquisition cost by 2--4.5x. | `02_inference/README.md` and `01_training_ada/transolver3_varpts/grad_nn/RESULTS.md`. | Needs a final timing table and hardware specification. |
| DrivAerML soft allocation improves point-budget error. | `dml_softlap_curve.png`; rAUEC 0.705 across 50 test cars. | Supported for the current pool/protocol; verify split and seeds. |
| AdaTransolver is faster at equal wall-clock time. | No consolidated equal-time comparison found. | Needs evidence; intentionally not claimed. |
| Gains are statistically significant. | Some runs use multiple seeds, but no consolidated paired intervals found. | Needs evidence; the draft avoids `significant`. |

## Main-text discipline audit

| Result | Class | Destination |
|---|---|---|
| Five-benchmark rAUEC table | Core discovery | Main text |
| NACA budget curve and kappa ablation | Necessary support | Main text |
| Replication identity for measure correction | Necessary support | Method/main text, one sentence |
| DrivAerML soft-allocation result and OOD diagnosis | Qualification/generalization | Main text |
| Learned acquisition head results | Supporting contribution | Main table and short paragraph |
| 21k error-network study | Failure boundary | Short main-text diagnostic; full table to appendix/SI |
| Density-adaptive training sweeps | Robustness/negative result | One main-text sentence; details to appendix/SI |
| Full parameter scans, per-sample plots, and extra policy variants | Provenance/robustness | Appendix or supplementary material |

## Author input and missing evidence

- Replace `Anonymous Authors` with the final author list only after authorship and affiliation approval.
- Confirm whether the paper name is `AdaTransolver` or another canonical project name.
- Re-export all three included raster figures with English labels and consistent fonts/colors.
- Consolidate the exact test sizes, seeds, checkpoint hashes, reconstruction settings, and hyperparameter-selection rule into one experiment registry.
- Add paired bootstrap confidence intervals for every main rAUEC value.
- Add one-pass uniform, two-pass uniform, equal-cumulative-point, and equal-wall-clock comparisons with full timing and memory decomposition.
- Verify every BibTeX record against the primary paper before submission, especially 2025--2026 arXiv and workshop-era references.
- Replace the bundled ICLR 2026 style with the official ICLR 2027 style when the official repository publishes it.

## Why this structure

- The introduction uses the `technical challenge -> insight -> contribution` variant because the key novelty is the sampling-measure mismatch, not the use of gradients alone.
- Results lead with the cross-benchmark evidence chain, then use NACA for mechanism-level detail and DrivAerML for an industrial-scale boundary case.
- Negative learned-error and density-adaptive-training results are compressed into diagnostic boundaries instead of competing with the main method.
- Claims are limited to the evidence currently present in the repository; missing latency and uncertainty results remain explicit submission blockers.
