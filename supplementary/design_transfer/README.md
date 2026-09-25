# Airfoil design comparisons

[Two-page PPT](latest_design_comparisons.pptx) · [Two-page PDF](latest_design_comparisons.pdf)

| Budget | Complete figure | Independent panels | Legends |
|---|---|---|---|
| 18.75% | [PNG](budget_01875.png) · [PDF](budget_01875.pdf) · [SVG](budget_01875.svg) | [Panels](budget_01875/panels/) | [Legends](budget_01875/legends/) |
| 37.5% | [PNG](budget_0375.png) · [PDF](budget_0375.pdf) · [SVG](budget_0375.svg) | [Panels](budget_0375/panels/) | [Legends](budget_0375/legends/) |

[Result table](comparison.csv) · [Full-precision summary](summary.json) · [Experiment settings](protocol.json)

These figures compare the completed 18.75% and 37.5% budget trials with the original and retuned density policies. They do not replace the selected trajectory in the main paper. Full mesh, previous AdaSolver, and updated AdaSolver use the same frozen operator, initial geometry, objective, and geometry constraints.

The updated policy changes the density exponent from 0.5 to 2 and the floor from 0.1 to 0.01. Both sparse variants in this transfer experiment use 563 coarse points and sampling seed 29, SLSQP with 80 maximum iterations and tolerance 1e-9, and fixed density throughout optimization. The published main-figure case instead used the historical 564-point coarse set and a 120-iteration limit. All six distinct initial/final geometries have converged Euler CFD with relative residual at most 1e-7.

Each figure shows the final CFD pressure, pressure prediction error on a shared scale, unamplified geometry changes, the recorded surrogate optimization trajectory, endpoint CFD quality versus median online time, and final Cd/Cl prediction error. CFD is available at the initial and final geometries only; the step curves show predictions. Times are medians of five paired repetitions with the same seed.

| Budget | Method | Online time (s) | Final CFD Cd/Cl | Cd/Cl error (%) |
|---|---|---:|---:|---:|
| 18.75% | Full mesh | 0.8376 | 0.05879236 | 3.261 |
| 18.75% | AdaSolver, previous | 0.3207 | 0.05874244 | 0.225 |
| 18.75% | AdaSolver, updated | 0.3804 | 0.05988042 | 1.626 |
| 37.5% | Full mesh | 0.8412 | 0.05879236 | 3.261 |
| 37.5% | AdaSolver, previous | 0.6603 | 0.05867482 | 3.771 |
| 37.5% | AdaSolver, updated | 0.4519 | 0.05881409 | 3.007 |

At 18.75%, the previous policy performs better than the updated policy in online time, final CFD objective, and coefficient prediction error. At 37.5%, the updated policy is 1.46× faster than the previous policy and reduces coefficient error, while its final CFD objective is 0.237% higher. Relative to full mesh at 37.5%, it is 1.86× faster, has slightly lower coefficient error, and reaches a final CFD ratio 0.037% higher. Field errors and full precision values are retained in `comparison.csv` and `summary.json`.

`latest_design_comparisons.pptx` contains two review slides with separately movable panels and native text. The full-resolution independent panels and legends are included in the two budget folders above, each in PNG, PDF, and SVG formats. The rendering source remains in the active local project at `airfoil_design_retuned/render_design.py`.
