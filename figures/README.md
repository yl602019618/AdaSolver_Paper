# Main-text figures

Figure 1 is exported from the first slide of the author-updated `总图1_density.pptx`. Figures 2–4 use their supplied author exports. Presentation content, typography, and layout are preserved. Each figure is cropped to visible content with a six-point margin. PNGs are rendered at 600 dpi; PDF text and lines remain vector graphics.

| Figure | PNG | PDF | Pixels |
| --- | --- | --- | --- |
| 1 | [PNG](fig1_overview.png) | [PDF](fig1_overview.pdf) | 5829 × 3806 |
| 2 | [PNG](fig2_field_comparison.png) | [PDF](fig2_field_comparison.pdf) | 10545 × 3373 |
| 3 | [PNG](fig3_airfoil_optimization.png) | [PDF](fig3_airfoil_optimization.pdf) | 8679 × 2986 |
| 4 | [PNG](fig4_superwing_optimization.png) | [PDF](fig4_superwing_optimization.pdf) | 8154 × 2927 |

The ten native Office equations are verified against the presentation and restored as LaTeX vectors where the Linux exporter omits them. The author's native layout and unrelated slide objects are preserved; Fig3/Fig4 density images and labels have been updated, with separately movable sampling overlays in the local source presentation.

Latest Fig3/Fig4 refresh: 2026-09-25. Airfoil and SuperWing use the actual policy density normalized to uniform = 1, with recorded sampling points. The shared linear colorbar spans 0–5. Inverse-design settings for these two figures are in Appendix B.5 and [the configuration record](../data/inverse_design_settings.json).

Airfoil pressure-policy refresh: Figure 2 retains the author layout and sample 1095, with regenerated CPG/CA sampling and signed-error images. The corresponding L2 annotations are 0.52% and 0.98%. The updated policy parameters and population results are in `data/airfoil_pressure/`.

## Additional design figures

The completed 18.75% and 37.5% design comparisons, including the two-page PPT, PDFs, individual panels, legends, and numerical results, are in [supplementary/design_transfer](../supplementary/design_transfer/README.md).

The final twelve-panel beta ablation is in Appendix D.1: [PDF](appendix/beta_ablation.pdf), [PNG](appendix/beta_ablation.png), and [independent panels](appendix/beta_ablation_assets/panels/). Its data and normalization are documented in [data/beta_ablation](../data/beta_ablation/README.md).
