# Exploratory seeds 1/2/3 numerical results

**Post-hoc exploratory subset.** Training seed 0 was excluded from this display after its test results were observed. It passed the original execution and protocol checks; the exclusion is not an invalid-run correction. The complete original seeds 0/1/2 data and summaries remain in the [original corrected package](../p0_airfoil_original_cdcl_20261117/README.md). The displayed seeds 1/2/3 therefore cannot be presented as an unselected three-seed stability validation.

| Method | Training seed 1 | Training seed 2 | Training seed 3 | Subset mean ± sample SD |
|---|---:|---:|---:|---:|
| PI | 0.310785 | 0.716833 | 0.909247 | 0.645622 ± 0.305520 |
| CPG | 0.303888 | 0.706133 | 0.767462 | 0.592494 ± 0.251815 |
| CA | 0.326603 | 0.731440 | 3.359502 | 1.472515 ± 1.646667 |

Within-checkpoint sampling variation is separate:

| Training seed | Method | Sampling seed 0 | Sampling seed 1 | Sampling seed 2 | Sampling mean ± SD |
|---:|---|---:|---:|---:|---:|
| 1 | PI | 0.178833 | 0.156525 | 0.596997 | 0.310785 ± 0.248118 |
| 1 | CPG | 0.166088 | 0.146940 | 0.598634 | 0.303888 ± 0.255437 |
| 1 | CA | 0.160122 | 0.177071 | 0.642615 | 0.326603 ± 0.273806 |
| 2 | PI | 0.649567 | 0.782278 | 0.718655 | 0.716833 ± 0.066375 |
| 2 | CPG | 0.625041 | 0.778945 | 0.714412 | 0.706133 ± 0.077286 |
| 2 | CA | 0.641575 | 0.713293 | 0.839453 | 0.731440 ± 0.100180 |
| 3 | PI | 1.699419 | 0.259735 | 0.768587 | 0.909247 ± 0.730076 |
| 3 | CPG | 1.530622 | 0.202842 | 0.568923 | 0.767462 ± 0.685794 |
| 3 | CA | 3.754263 | 0.233696 | 6.090548 | 3.359502 ± 2.948314 |
