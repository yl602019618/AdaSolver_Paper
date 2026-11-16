# Current Airfoil Cd/Cl result summary

The active three-training-seed Cd/Cl result is the [corrected original-policy package](data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/README.md). The historical P0 Cd/Cl rows use field-target policies and are archived diagnostics; they cannot be compared with the original Table 1 Cd/Cl policies.

| Method | Training seed 0 | Training seed 1 | Training seed 2 | Mean ± training sample SD |
|---|---:|---:|---:|---:|
| PI | 2.110458 | 0.310785 | 0.716833 | 1.046026 ± 0.943918 |
| CPG | 3.383230 | 0.303888 | 0.706133 | 1.464417 ± 1.673868 |
| CA | 2.041773 | 0.326603 | 0.731440 | 1.033272 ± 0.896537 |

The last column is sample SD across independent training checkpoints. [All sampling-seed results](data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/per_training_seed_summary.csv) are separate. The original control reproduces all 3,960 per-case errors exactly. All 11,880 new records, including the specified low-lift cases, are retained. The means above 1 do not show an average improvement over Uniform for these three models.
