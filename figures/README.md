# Main-text figures

Figures 1–4 use the author-provided exports from the
`car_elasticity_revision_20261117/paper/review_figures_tables` bundle.
The PDF and PNG files are copied without changing their content or layout.
The manuscript includes the PDFs; captions follow the accompanying READMEs.

| Figure | PNG | PDF | Pixels |
| --- | --- | --- | --- |
| 1 | [PNG](fig1_overview.png) | [PDF](fig1_overview.pdf) | 5804 × 3806 |
| 2 | [PNG](fig2_field_comparison.png) | [PDF](fig2_field_comparison.pdf) | 10545 × 3373 |
| 3 | [PNG](fig3_airfoil_optimization.png) | [PDF](fig3_airfoil_optimization.pdf) | 6350 × 2128 |
| 4 | [PNG](fig4_superwing_optimization.png) | [PDF](fig4_superwing_optimization.pdf) | 6083 × 2328 |

Figure 1 uses the final overview layout and aggregation formulas. Figure 2
updates the Elasticity sample 1600 PI signed-error panel (8.75%) and describes
the upper panels as refinement signals. Figures 3 and 4 use the author's final
left/right optimization layouts. Figure 3 shows displacement amplified 3×.

## Appendix updates

| Supplied figure | Paper asset | Change |
| --- | --- | --- |
| Fig5 | [Measure-correction ablation](appendix/beta_ablation.pdf) | Seven policy curves and deployed exponents |
| Fig6 | [Budget curves](appendix/budget_error_curves.pdf) | Seven policy curves at deployed β=1 |
| Fig11 | [Elasticity showcase](appendix/elasticity_fields.pdf) | Sample 1701 PI prediction/error and 5.51% annotation |
| RefinementScore | [Score overview](appendix/refinement_score_overview.pdf) | Eight-task score visualization |
| RefinementScore | [Score-to-points example](appendix/refinement_score_example.pdf) | Airfoil inputs, scores, weights and selected points |

The RefinementScore text and editable table are in
[`sections/refinement_score_visualization.tex`](../sections/refinement_score_visualization.tex).
The source table previews for Table1, Table2 and Table6 are integrated as editable
LaTeX in `tables/standard_benchmarks.tex`, `tables/industrial_benchmarks.tex` and
`tables/beta_ablation.tex`.

These author-provided PDF/PNG exports are the current figure sources. The full
precision summaries in `data/beta_ablation` and the ablation-table generator are
synchronized with the updated results. Existing SVG and independent-panel exports
for the replaced appendix figures predate this refresh.

Appendix C describes the supplied new Elasticity and ShapeNet-Car strategies.
Its compensation-exponent discussion has been removed; the ablation figures and
table retain the experimental exponent comparisons.

## Review versions

Run `python3 scripts/build_manuscript.py` from the repository root to refresh
both clean and marked PDFs. Changed graphics are framed in blue in the marked
paper, with baseline/current comparisons appended at the end. Keep
`review_assets/` with the marked TeX when sharing the review source.

## Additional design files

The existing design comparisons and individual panels remain documented in
[supplementary/design_transfer](../supplementary/design_transfer/README.md).
