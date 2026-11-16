# Airfoil Transolver3: original Cd/Cl policies across three training seeds

Status: **VERIFIED_COMPLETE**. This corrected package contains fresh evaluations of the three P0 Transolver3 checkpoints and their corresponding CA heads under the original Table 1 **Cd/Cl metric-specific policies**. It supersedes the Cd/Cl interpretation of the earlier P0 field-policy diagnostics. The earlier P0 numeric files remain available as provenance and retain their field-accuracy and timing scope.

The three new evaluation cohorts use independent training seeds **0, 1, 2**. Each contains the same **110 original cases × 3 sampling seeds × 3 budgets × 4 methods = 3,960 records**; the package contains **11,880 new records**, plus **3,960 fresh original-checkpoint control records**. PI and CPG use no CA head. CA uses the head trained for its own backbone.

Cd/Cl rAUEC is dimensionless and lower is better. Uniform is 1 for every checkpoint and sampling seed. Each training-seed value below is the mean of three sampling-seed rAUECs; the last column reports mean and sample SD (ddof=1) across the three independent training checkpoints.

| Method | Training seed 0 | Training seed 1 | Training seed 2 | Mean ± training sample SD |
|---|---:|---:|---:|---:|
| PI | 2.110458 | 0.310785 | 0.716833 | 1.046026 ± 0.943918 |
| CPG | 3.383230 | 0.303888 | 0.706133 | 1.464417 ± 1.673868 |
| CA | 2.041773 | 0.326603 | 0.731440 | 1.033272 ± 0.896537 |

The three-model means are above the Uniform reference of 1. The individual checkpoint results vary substantially; this experiment does not establish a reduction in mean Cd/Cl rAUEC across the three training seeds.

The population is exactly the original `protocol.json` list: reference pressure-force Cl > 0.02, with cases 1041 and 1139 excluded by the original threshold revision. No further case or outlier was removed. The budgets are 0.375, 0.50 and 0.75; sampling seeds are 0, 1 and 2. Policies are `pi_cdcl_tuned`, `cpg_cdcl_budget10` and `ca_cdcl_budget10`, with every original scientific parameter retained. CPG uses 1,127 coarse points (10% cap); CA uses the original 563 coarse points (5%). All three P0 checkpoints share their original P0 normalizer and pressure target channel 3.

For each checkpoint, sampling seed and method, we first calculate the 110-case mean error at each budget. We integrate those three values with the trapezoidal rule and divide by the Uniform area from the same checkpoint, sampling seed, test population and reconstruction. We then average the three sampling-seed ratios. Full 100% is excluded from the integral. Every record uses the original `index_grid_linear_nearest` reconstruction and original pressure-force integration.

The original frozen checkpoint is a separate control, with its original normalizer. All 3,960 control case errors match the published original CSV exactly; its Table 1 values and sampling SDs are:

| Method | Original Table 1 | Fresh original-checkpoint control |
|---|---:|---:|
| PI | 0.2174079 ± 0.1687530 | 0.2174079 ± 0.1687530 |
| CPG | 0.2379492 ± 0.1781006 | 0.2379492 ± 0.1781006 |
| CA | 0.2452788 ± 0.1779119 | 0.2452788 ± 0.1779119 |

The original control was trained under the historical 1000/200 protocol, while the three P0 models use the matched 900/100 training/validation protocol and their common fitted normalizer. The original control and P0 seed0 are distinct checkpoints. [Their separate summaries](original_control_vs_new_seed0.csv) are descriptive, not paired model deltas.

The supplied `official/scripts/evaluate_airfoil2d_cdcl_official.py` is an archived 25% single-feature candidate. The source that generated the specified Table 1 records is `scripts/airfoil2d/run_budget10.py` with `adasolver.evaluation.airfoil2d.AirfoilEvaluator`, followed by the original aggregation in `airfoil_pi_tuning/analyze.py`. This package uses those native evaluation functions unchanged. The discrepancy, exact control reproduction and source hashes are recorded in `verification.json` and `source_hashes.json`.

Airfoil Cd/Cl is sensitive to small predicted lift. For case 1198, training seed0 / CPG / sampling seed1 / budget0.375 predicts Cl=0.0003855034 and has Cd/Cl absolute error 51.284231. For the same case, training seed1 / Uniform / sampling seed1 / budget0.375 predicts Cl=0.0002174781 and has error 88.054804. The latter also enlarges that sampling run's Uniform denominator. Both method errors and paired denominators therefore contribute to the variation. These observed cases remain in all summaries.

| Case | Reference Cl | Reference Cd/Cl | Included |
|---:|---:|---:|---|
| 1007 | 0.0241763662 | 1.4460974942 | Yes |
| 1059 | 0.0246764336 | 1.2771278323 | Yes |
| 1198 | 0.0244128776 | 0.8084258134 | Yes |

All NaN/Inf, case-count, policy, checkpoint/head, normalizer and reconstruction checks passed. `case_1007_1059_1198.csv` preserves all 432 control/new-model records for the three specified cases. Sampling SD and training SD are stored separately.

| File | Contents |
|---|---|
| [protocol_original_cdcl.json](protocol_original_cdcl.json) | Original policies, case list, budgets, sampling seeds and aggregation definition |
| [case_filter.csv](case_filter.csv) | Reference forces and inclusion decisions for original IDs 1000–1199 |
| [per_case_records.csv.gz](per_case_records.csv.gz) | 11,880 new per-case records, forces, errors, point-set hashes and asset bindings |
| [per_seed_sampling_summary.csv](per_seed_sampling_summary.csv) | 108 method/training/sampling/budget rows, including both areas and rAUEC |
| [per_training_seed_summary.csv](per_training_seed_summary.csv) | Each checkpoint's sampling-seed values, mean and sampling SD |
| [training_seed_cdcl_summary.csv](training_seed_cdcl_summary.csv) | Three independent checkpoint values and training mean/sample SD |
| [uniform_denominators.csv](uniform_denominators.csv) | All nine checkpoint/sampling-seed Uniform areas |
| [checkpoint_registry.csv](checkpoint_registry.csv) | Checkpoint/head paths and hashes, training/head seeds, normalizer and reconstruction |
| [verification.json](verification.json) | Complete checks, control gate, resolved entrypoint discrepancy and execution-check history |
| [source_hashes.json](source_hashes.json) | Native source, references, analysis scripts and launch receipts |
| [reproduction_command.md](reproduction_command.md) | CSV-only validation and exact GPU rerun commands |
| [RESULTS_DETAIL.md](RESULTS_DETAIL.md) | Full sampling/training breakdown and case diagnostics |
| [reviewer_reply.md](reviewer_reply.md) | English response based on this corrected package |

This package evaluates Transolver3 Cd/Cl accuracy. New formal latency for these metric-specific policies, LinearNO under the original Cd/Cl policies, and Plasticity three-training-seed experiments are outside its completed scope. The earlier P0 field-policy timings are not relabeled as timings of the corrected Cd/Cl policies.
