# Airfoil pressure and force-ratio results

Table 1 and the appendix budget figure use the same 110 test geometries with reference pressure-force Cl > 0.02, seeds 0, 1, and 2, and budgets 37.5%, 50%, and 75%. This is a post-hoc threshold change from Cl > 0.01 (112 cases), removing IDs 1041 and 1139. The earlier field results are preserved in `history/cl_gt_0p01/`.

For each seed, the method's mean-error curve is integrated by the trapezoidal rule and divided by the paired Uniform integral. Table 1 reports the mean and sample standard deviation of these three rAUECs. This differs from integrating the pooled mean curve. `budget_summary.csv` gives both raw and full-mesh-normalized errors for all 24 plotted points.

- Field: `configs.json`, `test_records.csv`, `test_summary.json`.
- Absolute Cd/Cl error: `cdcl_configs.json`, `cdcl_test_records.csv`, `cdcl_test_summary.json`.
- Same-backbone full-mesh normalization: `fullmesh_records.csv`.
- Cohort and selection: `protocol.json`, `pi_frozen_selection.json`.
- PI confirmation: `pi_confirmation_summary.json` uses the existing 48-geometry confirmation pool and seeds 6, 7, and 8. The field-only `confirmation_records.csv` retains the earlier CPG/CA confirmation run.
- `recompute.py` verifies every reported seed statistic and figure ordinate from the compact records.

PI now uses the same geometry-only policy for both metrics: amplitude 20, surface-distance scale 0.3, and index-grid Voronoi compensation with beta 1. It retains 121 surface nodes. Coefficient PI selection screened 127 candidates on 24 calibration geometries and selected one setting on 46 separate calibration geometries with three seeds. All budgets use this common setting. The test results did not select it.

The CPG and CA field policies are unchanged from their earlier selection under Cl > 0.01. CPG uses a squared normalized pressure-gradient weight with a 0.1 floor; CA uses a 0.01 floor and nested sampling. Both use 10% coarse inputs, 16 gradient neighbors, and beta 1. The pressure head remains 128→8→32→1 with 1,353 parameters. CPG and CA retain their separate coefficient policies.

The available original full-mesh-trained Airfoil checkpoint predicts Mach (Q channel 4), whereas this task predicts pressure (Q channel 3). Its field errors and pressure-force integration are not comparable. Those eight old plotted points are retained in `history/invalid_mach_reference.csv` and excluded from the current figure. `reference_target_audit.json` records the target check. The horizontal full-mesh reference uses the matching pressure model and the current 110 cases.

Timing measurements in `timing_raw.jsonl` use the field policies, which are unchanged by this update. Timing remains table-only in the manuscript. Inverse-design settings and CFD results are a separate experiment.

The appendix now includes a completed full-mesh-trained pressure Transolver3 reference for Airfoil at all four budgets. See `../airfoil_fullmesh_reference/` for its training metadata and 1,100 evaluation records. The historical Mach checkpoint remains excluded.
