# Claim card

Supported implementation statements: native point-temp accepts variable point sets; its key-mass correction passes unit-weight and explicit-duplication checks; original conv-temp rejects arbitrary point sets and supports the full grid; replayed common sampling RNG states match saved training checkpoints. These checks do not establish accuracy or speed improvement.

Trained accuracy is accepted only for rows marked true in REPORT.md and backed by results/{architecture}_accuracy_acceptance.json. No checkpoint is selected with test error. The CPG beta=0/1 mechanism keeps coarse and selected index hashes paired.

Observed pressure-field results in the fixed 110-case population:

- LinearNO_point_temp: CPG beta=1 rAUEC 0.559296 (44.07% less area than its matched Uniform); paired beta=0 rAUEC 6.201754.
- LinearNO_point_temp: CA rAUEC 0.589981; its observed mean is higher than CPG beta=1. This ordering is retained.
- Transolver3: CPG beta=1 rAUEC 0.529345 (47.07% less area than its matched Uniform); paired beta=0 rAUEC 7.901202.
- Transolver3: CA rAUEC 0.563249; its observed mean is higher than CPG beta=1. This ordering is retained.

This beta=0/1 experiment supports the compensation mechanism for this cohort and field-target policy. It does not identify the optimum over untested beta values.

Absolute Full errors (same 110-case population): Transolver3=0.004092686, LinearNO_point_temp=0.004775013, LinearNO_conv_temp_fullgrid=0.002791744.

The convolutional variant uses its native full-grid training/input path and has no subsampling/rAUEC result. Its Full accuracy must remain visible when interpreting the point-variant comparison.

Complete-request timing supported: False. Reconstruction-only speedups and cache reuse have separate meanings from A-scope end-to-end speedups.

Comparative claims about which backbone is faster or more accurate must be drawn from completed absolute field-error and same-scope latency rows. Each backbone has its own Uniform rAUEC denominator; those normalized scores cannot rank absolute model accuracy. Unfavorable results are preserved.
