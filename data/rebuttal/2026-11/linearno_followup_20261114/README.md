# Verified reuse subset of the three-training-seed follow-up

This package rechecks existing ShapeNet-Car E5/E3 results from paper release
`91c1fa7cc81b8c1cb61a43618296c1c892d818c0`. The requested three-dataset follow-up
is **partial**. New Darcy/Airfoil cohorts, missing Car LinearNO seeds and heads,
and the final old/new complete-output latency matrix are pending.

The local audit indexed9,689 relevant files and checked existing experiment
controllers before scheduling new work. Three E5 Transolver3 model/head pairs
and E3 LinearNO seed0 were reused. No result was fabricated for a missing cell.
The existing seed0 workflows retain ownership of their outputs.

Raw-record replay checked22,200 E5 metric rows and11,200 E3 metric rows, rehashed
the model/head/normalizer assets, and recomputed case/sampling/training summaries.
E3 latency summaries were recomputed from276 conditions with100 formal repeats
and20 warmups per condition. The reconstruction implementation gate contains108
real fixed-prediction parity rows; it does not replace new-checkpoint request
validation. Detailed numbers and input hashes are in verified_reuse.json.

E5 field CPG/CA use beta1. E5 Cd CPG/CA use beta0.75 and remain explicitly labelled legacy configurations.
The requested beta1 constraint applies to CPG; CA retains its recorded frozen beta. E3 and E5 Transolver3 seed0 overlap and are not independent
training models. Existing E1 Car timing uses another checkpoint and is not paired
with these errors. Historical training Git revisions and actual GPU busy seconds
remain missing where the original records did not record them; reported training
cost is the recorded process wall time, with its scope retained in the registry.

For each model and sampling seed, rAUEC integrates case-average errors at
0.375/0.50/0.75 and divides by matched Uniform area. Full100% is outside the
integral. Training variation is sample SD over three independently trained models;
sampling SD and timing-repeat SD have separate fields. Each backbone uses its own
denominator. Absolute errors provide the cross-model accuracy comparison.

Darcy retains disclosure of historical test visibility. The running900/100
physical-loss cohort differs from the separate1000/200 normalized-loss cohort;
their seeds cannot be pooled. Airfoil retains the published110 positive-lift
cases and221×51 grid. See protocol_differences.md and pending.json.

Weights, raw fields, private conversations and long logs remain external. The
relative raw-record index and hashes identify internal source records. Official
LinearNO comes from https://github.com/HiPRL/LinearNO at
`3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269`; no license was present at that pin,
so its implementation is not redistributed in this package.

From the repository root, run:

```sh
python3 reproducibility/rebuttal/2026-11/linearno_followup_20261114/validate_release.py --repo .
```

The validator recomputes the public rAUEC and training mean/sample SD, checks
schema/identity constraints, decodes PNGs and verifies the release manifest.
Rendering uses the accompanying deterministic plot script and these CSVs.
The header-only reconstruction_summary.csv is explicitly pending, not a measured
zero-cost reconstruction result. The full requested package is not yet complete.
