# AdaSolver paper

## Standard benchmarks: three independent training seeds

This branch publishes the completed Elasticity, Pipe and Darcy training-seed
supplement: **9 independently trained backbones, 9 matching CA heads and 66,600
test observations**. Results report mean and sample SD across three independent
training runs; inference-sampling variability is provided separately.

**本分支内容：Elasticity / Pipe / Darcy 的三独立训练种子补充实验。**
完整训练、评估和数值核验均已完成。请从
[结果说明与文件索引](STANDARD_TRAINING_SEEDS.md)查看核心表格、逐 seed 结果、
统计口径和复现入口。

- [Complete results and tables](data/rebuttal/2026-11/E5_standard/README.md)
- [Per-training-seed rAUEC](data/rebuttal/2026-11/E5_standard/training_seed_rauec.csv)
- [Configurations and reproduction](reproducibility/rebuttal/2026-11/E5_standard/README.md)

## Manuscript

[Compiled paper](paper.pdf) · [LaTeX source](iclr2027_conference.tex)

Build with TeX Live (`latexmk`, `pdflatex`, BibTeX), Python 3, and Poppler
(`pdfinfo`, `pdftotext`):

```sh
python3 scripts/build_manuscript.py
```

The build updates `paper.pdf` and keeps temporary files in `.build/`. It preserves
the manuscript text, supplied figures, and result tables.

The `v1/` directory contains earlier paper material. Current editing starts from
`iclr2027_conference.tex` and its included sections.

## Main figures

Figures 1--4 use the author-edited exports in [figures/](figures/README.md).
Their PDF and PNG contents are preserved; LaTeX includes the vector PDFs.
Captions and labels are defined in the four `figures/fig*.tex` files.

The actual density fields and retained sampling points in Figures 3 and 4 are
documented in [data/design_density](data/design_density/README.md), with compact
source arrays and an independent panel renderer.

The inverse-design protocols for Figures 3 and 4 are described in Appendix B.5
and recorded in [data/inverse_design_settings.json](data/inverse_design_settings.json).

## Figures and design downloads

| Content | Files |
|---|---|
| Fig1–4 | [All figures](figures/README.md) |
| Appendix D.1: measure-correction ablation | [Figure PDF](figures/appendix/beta_ablation.pdf) · [PNG](figures/appendix/beta_ablation.png) · [Data and reproduction](data/beta_ablation/README.md) |
| Fig3: Airfoil design | [PNG](figures/fig3_airfoil_optimization.png) · [PDF](figures/fig3_airfoil_optimization.pdf) |
| Fig4: SuperWing design | [PNG](figures/fig4_superwing_optimization.png) · [PDF](figures/fig4_superwing_optimization.pdf) |
| Additional design comparisons | [Figures, PPT, panels, and results](supplementary/design_transfer/README.md) |
| Current inverse-design settings | [LaTeX](sections/inverse_design_settings.tex) · [Parameters](data/inverse_design_settings.json) |
