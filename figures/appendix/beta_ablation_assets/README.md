# β ablation overview

All metrics complete: True.

Figure style follows the supplied appendix budget figure. Both axes use a metric name followed by its mathematical symbol: "Compensation exponent β" and "Error–budget area ratio R_m(β)". For policy m and seed s, let A_m,s(beta) be the trapezoidal error-budget area over 37.5%, 50%, and 75% budgets. The displayed value is R_m(beta) = mean_s[A_m,s(beta) / A_m,s(0)], so the denominator is the same policy and seed at beta=0. Bands show one sample standard deviation of these ratios across the three sampling seeds. The dashed reference is no measure correction; gray rings mark the unchanged paper beta.

All twelve metrics use their complete planned test populations. There are no provisional panels.

| Dataset | Metric | Status | Cases | PI rAUEC | CPG rAUEC | CA rAUEC |
|---|---|---|---:|---:|---:|---:|
| Airfoil2D | field | complete | 110/110 | 0.6166 | 0.5831 | 0.5579 |
| Pipe | field | complete | 200/200 | 0.8711 | 0.6719 | 0.7051 |
| Darcy | field | complete | 200/200 | 0.9965 | 0.6419 | 0.6400 |
| Elasticity | field | complete | 200/200 | 0.7280 | 0.4681 | 0.5469 |
| Plasticity | field | complete | 80/80 | 0.8888 | 0.8890 | 0.8897 |
| ShapeNet-Car | field | complete | 100/100 | 0.5895 | 0.5577 | 0.5577 |
| DrivAerML | field | complete | 50/50 | 0.8834 | 0.8246 | 0.8263 |
| SuperWing | field | complete | 2871/2871 | 0.7564 | 0.6706 | 0.7193 |
| Airfoil2D | cdcl | complete | 110/110 | 0.2174 | 0.2379 | 0.2453 |
| ShapeNet-Car | cd | complete | 100/100 | 0.6561 | 0.6474 | 0.6479 |
| DrivAerML | cd | complete | 50/50 | 0.9245 | 0.9209 | 0.9244 |
| SuperWing | cdcl | complete | 2563/2563 | 0.8246 | 0.8026 | 0.8167 |

The table uses rAUEC relative to matched Uniform at the paper beta; it has a common reference across methods. The figure uses each policy’s own beta=0 reference.

Files: `Fig_beta_ablation_overview.png/pdf/svg`; `panels/` and `legends/` contain individual assets; `data/summary.json/csv` contains the displayed values.

Reproduce this figure with python3 scripts/render_beta_ablation.py from the repository root.
