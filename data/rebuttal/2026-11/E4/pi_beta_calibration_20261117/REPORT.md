# Darcy PI β calibration

Status: final test matrix verified.

The original E4 checkpoint, normalizer, PI prior and reconstruction rule are retained. Only the compensation exponent changes. All beta variants for one case/budget/sampling seed share the same selected-index hash. The disjoint calibration cohort is smooth2 rows400–499 (IDs1400–1499); the final E4 cohort is rows0–199 (IDs1000–1199). The original checkpoint and historic policies had test-informed development, so this is not described as a never-viewed holdout.

Candidates: 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.5, 2. Exponents1.5 and2 lie above1; the whole search is not called0–1 tempering.

The primary shared beta is **1**, selected by minimizing the worst resolution’s validation mean rAUEC. Ties within1e-12 choose the value closest to the original0.5, then the lower beta. Per-resolution minima are secondary validation-selected diagnostics. Final-test values do not choose beta or select between these two policies.

rAUEC first averages case errors separately for each sampling seed and budget, then forms the three-budget trapezoid ratio to the same resolution’s Uniform, then reports mean and sample SD (ddof=1) over sampling seeds0/1/2. Full100% is excluded from the integral. The experiment uses one frozen training model; sampling SD is not training SD.

Observed final primary result: shared-beta mean rAUEC below1 at all three resolutions = **True**. This status concerns the three means; individual sampling-seed values remain in test_rauec.csv.

| Resolution | Original beta0.5 | Shared beta | Shared rAUEC ± sampling SD | Secondary validation beta | Secondary rAUEC ± sampling SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 85² | 0.996622 ± 0.018057 | 1 | 0.998995 ± 0.019993 | 0.625 | 0.993774 ± 0.018484 |
| 141² | 1.000776 ± 0.005778 | 1 | 0.971548 ± 0.003287 | 1.5 | 0.970917 ± 0.000803 |
| 211² | 1.003448 ± 0.013816 | 1 | 0.969888 ± 0.012662 | 2 | 0.959246 ± 0.010756 |

All validation candidates and all prespecified test conditions are retained, including values at or above1 and regressions relative to beta0.5. No test-best beta is computed. The old published E4 table is a separate fixed-beta result and is not overwritten by this calibration study.

Files: `selection.json`, `results/validation_rauec.csv`, `results/validation_candidate_objectives.csv`, `results/test_rauec.csv` and `results/absolute_errors.csv` when complete. Raw observations are retained in `records/`; matrix and identity checks are in the statistical receipts. No formal latency conclusion follows from these accuracy runs.

Public reproduction: `python3 verify_published.py --package . --output /tmp/darcy-pi-beta-check.json`. Raw records are lossless gzip; the verifier materializes the original CSV bytes in a temporary directory and uses the unchanged statistics source.
