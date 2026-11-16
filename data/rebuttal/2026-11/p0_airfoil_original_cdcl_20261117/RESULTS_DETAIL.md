# Corrected Airfoil Cd/Cl results

Original metric-specific policies; 110 cases; three budgets; paired Uniform normalization. Training values average the three sampling rAUECs.

| Method | Training seed 0 | Training seed 1 | Training seed 2 | Mean ± training sample SD |
|---|---:|---:|---:|---:|
| PI | 2.110458 | 0.310785 | 0.716833 | 1.046026 ± 0.943918 |
| CPG | 3.383230 | 0.303888 | 0.706133 | 1.464417 ± 1.673868 |
| CA | 2.041773 | 0.326603 | 0.731440 | 1.033272 ± 0.896537 |

The following SDs measure sampling variability within each trained checkpoint.

| Training seed | Method | Sampling seed 0 | Sampling seed 1 | Sampling seed 2 | Sampling mean ± sampling SD |
|---:|---|---:|---:|---:|---:|
| 0 | PI | 1.249107 | 2.560888 | 2.521380 | 2.110458 ± 0.746214 |
| 0 | CPG | 1.339347 | 6.082938 | 2.727405 | 3.383230 ± 2.438851 |
| 0 | CA | 1.076556 | 2.874901 | 2.173862 | 2.041773 ± 0.906420 |
| 1 | PI | 0.178833 | 0.156525 | 0.596997 | 0.310785 ± 0.248118 |
| 1 | CPG | 0.166088 | 0.146940 | 0.598634 | 0.303888 ± 0.255437 |
| 1 | CA | 0.160122 | 0.177071 | 0.642615 | 0.326603 ± 0.273806 |
| 2 | PI | 0.649567 | 0.782278 | 0.718655 | 0.716833 ± 0.066375 |
| 2 | CPG | 0.625041 | 0.778945 | 0.714412 | 0.706133 ± 0.077286 |
| 2 | CA | 0.641575 | 0.713293 | 0.839453 | 0.731440 ± 0.100180 |

The original frozen checkpoint is a separate control. Both columns below use sampling SD.

| Method | Original Table 1 | Fresh original-checkpoint control |
|---|---:|---:|
| PI | 0.2174079 ± 0.1687530 | 0.2174079 ± 0.1687530 |
| CPG | 0.2379492 ± 0.1781006 | 0.2379492 ± 0.1781006 |
| CA | 0.2452788 ± 0.1779119 | 0.2452788 ± 0.1779119 |

All required diagnostic cases are retained.

| Case | Reference Cl | Reference Cd/Cl | Included |
|---:|---:|---:|---|
| 1007 | 0.0241763662 | 1.4460974942 | Yes |
| 1059 | 0.0246764336 | 1.2771278323 | Yes |
| 1198 | 0.0244128776 | 0.8084258134 | Yes |

Complete per-case errors: [case_1007_1059_1198.csv](case_1007_1059_1198.csv). Denominator variation and largest errors: [variance_diagnostics.json](variance_diagnostics.json).
