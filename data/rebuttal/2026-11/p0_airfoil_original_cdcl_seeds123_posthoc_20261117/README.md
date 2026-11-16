# Airfoil Cd/Cl: exploratory display of training seeds 1/2/3

**Post-hoc exploratory subset.** Training seed 0 was excluded from this display after its test results were observed. It passed the original execution and protocol checks; the exclusion is not an invalid-run correction. The complete original seeds 0/1/2 data and summaries remain in the [original corrected package](../p0_airfoil_original_cdcl_20261117/README.md). The displayed seeds 1/2/3 therefore cannot be presented as an unselected three-seed stability validation.

Seed 3 was fixed before its training and test evaluation. It uses the same P0 Transolver3 architecture, 900/100 split, normalizer, 1,000 training epochs and 113,000 updates. Its own pressure head was trained for 200 epochs. No further seed was selected according to seed 3 performance.

The table uses original Cd/Cl-specific policies, all 110 original cases, sampling seeds 0/1/2, and budgets 37.5%, 50%, 75%. CPG uses 10% coarse sampling and CA retains the original 5%. Original reconstruction and pressure-force integration are unchanged. Uniform rAUEC is1.

| Method | Training seed 1 | Training seed 2 | Training seed 3 | Subset mean ± sample SD |
|---|---:|---:|---:|---:|
| PI | 0.310785 | 0.716833 | 0.909247 | 0.645622 ± 0.305520 |
| CPG | 0.303888 | 0.706133 | 0.767462 | 0.592494 ± 0.251815 |
| CA | 0.326603 | 0.731440 | 3.359502 | 1.472515 ± 1.646667 |

CA has a subset mean rAUEC of 1.472515, above the Uniform reference of 1. Its seed 3 value is 3.359502. For case 1198 at the 50% budget and sampling seed 2, CA predicts Cl=0.0002118407 against reference Cl=0.0244128776, giving a Cd/Cl absolute error of 92.266864. This error remains in the summaries.

These SDs describe the three displayed checkpoints. They are not estimates from an unselected training-seed sample. Each checkpoint value first averages three sampling-seed rAUECs. Each sampling ratio uses its own checkpoint/sampling-seed Uniform area after case averaging and trapezoidal integration. Full 100% is excluded.

Seed1/2 contribute 7,920 records reused from their already verified **original-Cd/Cl** evaluations, and seed 3 contributes 3,960 fresh records. No P0 field-policy force diagnostics are used. The separate original frozen-checkpoint control is reused with its exact3,960-record reproduction evidence. All110 cases, including1007/1059/1198, remain; exclusions1041/1139 follow the original Cl>0.02 protocol.

Files: [per-case records](per_case_records.csv.gz), [new seed 3 records](seed3_per_case_records.csv.gz), [per-budget/sampling table](per_seed_sampling_summary.csv), [per-checkpoint sampling SD](per_training_seed_summary.csv), [training subset table](training_seed_cdcl_summary.csv), [Uniform denominators](uniform_denominators.csv), [checkpoint/head registry](checkpoint_registry.csv), [case diagnostics](case_1007_1059_1198.csv), [verification](verification.json), [protocol](protocol_original_cdcl.json), [source hashes](source_hashes.json), [reproduction](reproduction_command.md), and [English response](reviewer_reply.md).

The completed scope is Transolver3 original-Cd/Cl accuracy for this disclosed exploratory subset. This package makes no completion claim for Plasticity three-training-seed experiments or new formal latency of the Cd/Cl policies.
