# Measure-correction ablation

The final evaluation covers eight benchmarks, twelve metrics, beta values 0, 0.25, 0.5, 0.75 and 1, sampling seeds 0/1/2, and retained fractions 37.5%, 50% and 75%. Coarse/final points, model and head checkpoints, and reconstruction are fixed within each sampling policy. The exponents used in the main experiments are retained.

- `summary.json` contains complete-population results and the per-seed, per-budget values.
- `summary.csv` contains every policy/beta aggregate. Its `rauec_mean` uses matched Uniform as the denominator; `relative_to_beta0_mean` uses each policy and seed's beta=0 error-budget area before averaging seeds.
- `protocol.json` defines the evaluated grid, populations, fixed variables and metric references.
- `airfoil_ratio_case.json` contains the saved coefficient and pressure-error values for the ratio-sensitivity example in Appendix D.1.
- `table_annotations.csv` records the 36 within-policy comparisons. Parenthetical percentages in Table 4 are `100 * (1 - selected_mean_rauec / beta0_mean_rauec)`, calculated before rounding. These compare the table's means; the figure separately averages the per-seed ratios to beta=0.

The paper-setting results reproduce all 36 corresponding entries in the main result tables at their displayed precision. No missing or nonfinite test cases were dropped. The figure bands are sample standard deviations across three sampling seeds, not confidence intervals over geometries.

Render the figure and its independent panels from the repository root:

```sh
python3 scripts/render_beta_ablation.py --full-only
python3 scripts/build_beta_ablation_table.py
```

The main output is `figures/appendix/beta_ablation.pdf`, with PNG/SVG versions. Independent panels, the shared legend and plotting checks are in `figures/appendix/beta_ablation_assets/`.
