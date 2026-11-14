# LinearNO standard-task figures

The deliverables are local ICLR-style result tables and quantitative figures generated from the new cohort. They do not change the current manuscript.

The evidence questions are whether adaptive sampling improves each backbone relative to its own Uniform baseline, how the backbones compare in absolute field error, how beta correction changes the fixed-point-set CPG result, and how full-output latency changes under the same reconstruction rule.

Sources are `results/field_records.csv`, the independently rebuilt budget/rAUEC tables, and the accepted condition-level latency CSV. The collector binds each row to the frozen checkpoint, train-only normalizer, fixed population, sampling seed, strategy target, beta and reconstruction ID. `results/collection_status.json` distinguishes a preliminary complete cohort from the finished experiment matrix. No table value comes from initialized models or engineering fixtures.

| Artifact | Content and grouping | Uncertainty and reference |
| --- | --- | --- |
| rAUEC table | Rows grouped by Transolver3 and official LinearNO point-temp; methods Uniform, PI, CPG, CA. Columns Darcy Field and Airfoil Field. Separate native/optimized reconstruction tables. | Mean above sample SD across three sampling seeds; best/second within each backbone, dataset and protocol only. CPG beta=1. |
| rAUEC figure | Darcy and Airfoil panels; each panel contains the two backbone groups with consistent method colors. | Uniform=1 reference. Each backbone uses its own Uniform denominator. |
| CPG mechanism table | Paired beta=0/1 rAUEC plus seedwise differences. | Same coarse prediction and selected indices, verified upstream. |
| Absolute error curves | Shared field-error axis for both backbones within each dataset; 37.5/50/75% budgets. | Sampling SD; separate horizontal Full errors. |
| Efficiency and reconstruction comparison | Condition-level median/p95 and reconstruction profiles; old/new within scope A; scopes B/C separate. | All 100 direct repeats retained; plots label any median of condition summaries. |
| Absolute error versus latency | Same complete population accuracy curves, latency from the three fixed timing cases; native/optimized reconstruction in separate panels. | Field error is not rAUEC. Full remains an independent absolute reference. |

Use the existing `paper_git/tables/standard_benchmarks.tex` booktabs layout as the table style reference, with no historical values copied. Methods keep the same colors across figures: Uniform gray, PI orange, CPG blue, CA green. Backbone differences also use line styles/markers, so color is not the only cue. Native conv-temp is a separate Full-only reference and has no rAUEC. ShapeNet-Car will only join after its comparable reevaluation; otherwise its legacy results remain separate.

Each plot is exported as PDF, SVG with live text, and 300 dpi PNG. CSVs and plotting code remain adjacent in the archive. Missing methods are labeled N/A with the current reason; they are never assigned zero. Final rendered QA checks clipping, readable labels, distinct line styles, and exact source-number correspondence. No best-backbone claim is inferred from separately normalized rAUEC values.
