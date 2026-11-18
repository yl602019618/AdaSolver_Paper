# Measure-correction ablation

The evaluation covers eight benchmarks, twelve metrics, beta values 0, 0.25, 0.5, 0.75 and 1, sampling seeds 0/1/2, and retained fractions 37.5%, 50% and 75%. Coarse/final points, model and head checkpoints, and reconstruction are fixed while beta varies within each sampling policy.

The current data incorporates the supplied `car_elasticity_revision_20261117`
update: Elasticity PI Field and all three ShapeNet-Car policies for Field and
Cd. These seven curves contribute 35 updated beta points, each retaining its
per-seed and per-budget values; the other 145 points are unchanged. All seven
updated curves deploy beta = 1. The complete-precision source is
`figure_table_sources/figures/Fig5/updates/summary_after_update.json` within that
bundle. Its selected endpoints agree with
`figure_table_sources/tables/Table6/updates/beta_ablation.csv`.

- `summary.json` contains complete-population results and the per-seed, per-budget values.
- `summary.csv` contains every policy/beta aggregate. Its `rauec_mean` uses matched Uniform as the denominator; `relative_to_beta0_mean` uses each policy and seed's beta=0 error-budget area before averaging seeds.
- `protocol.json` defines the evaluated grid, populations, fixed variables and metric references.
- `airfoil_ratio_case.json` contains the saved coefficient and pressure-error values for the ratio-sensitivity example in Appendix D.1.
- `table_annotations.csv` records the 36 within-policy comparisons. Parenthetical percentages in the measure-correction table are `100 * (1 - selected_mean_rauec / beta0_mean_rauec)`, calculated before rounding. These compare the table's means; the figure separately averages the per-seed ratios to beta=0.

The paper-setting results reproduce all 36 corresponding entries in the main result tables at their displayed precision. No missing or nonfinite test cases were dropped. The figure bands are sample standard deviations across three sampling seeds, not confidence intervals over geometries.

Render the figure and its independent panels from the repository root:

```sh
python3 scripts/render_beta_ablation.py --full-only
python3 scripts/build_beta_ablation_table.py
```

The main output is `figures/appendix/beta_ablation.pdf`, with PNG/SVG versions.
Independent panels, the shared legend and plotting checks are in
`figures/appendix/beta_ablation_assets/`. Gray rings labeled `Deployed beta` use
each row's `production_beta`. The table generator updates
`tables/beta_ablation.tex` and `table_annotations.csv` from the same full-precision
summary. The ordinary manuscript build uses the existing figure and table files.
