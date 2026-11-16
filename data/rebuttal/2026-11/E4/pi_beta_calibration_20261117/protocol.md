# Darcy PI beta calibration

Only the compensation exponent changes. The original E4 checkpoint, global train-derived normalizers, coordinate precision, flattened-EDT prior, weighted-without-replacement selection, and linear/nearest reconstruction remain fixed. All betas share each actual retained set.

Calibrate on100 smooth2 rows400–499 (global IDs1400–1499), disjoint from the200 E4 evaluation cases. A common beta is selected from the fixed11-value grid by minimum worst-resolution validation rAUEC; a separate resolution-calibrated variant is selected within each resolution. Ties within1e-12 prefer beta closest0.5, then the lower value. The final evaluation includes both validation-selected choices and beta0.5, without changing choices after observing test results.

Each sampling seed first averages errors over cases at37.5/50/75%, integrates that curve and divides by its matching Uniform area; the three ratios yield mean and sample SD. Full is excluded. The three seeds are sampling replicates of one frozen training model.

The old checkpoint and published E4 policies have test-informed history. This calibration uses a disjoint population but does not claim a globally untouched holdout. Beta1.5/2 are explicitly over-correction candidates. Weighted-without-replacement nominal proposal correction is not an exact final inclusion-probability correction. No below-one guarantee or monotonic improvement is assumed.

Final matrices retain original beta0.5 and all validation candidates, even if the requested below-one criterion is not achieved. This experiment is separate from the locked published E4 result and makes no formal speed claim.
