# AdaSolver paper

<!-- airfoil-seeds123-posthoc -->
**Exploratory display: training seeds 1/2/3.** Seed 0 was excluded after its test performance was observed, while its valid original data remain preserved. [Subset results](data/rebuttal/2026-11/p0_airfoil_original_cdcl_seeds123_posthoc_20261117/README.md) · [Displayed values](RESULTS_DETAIL.md) · [English response](reviews/airfoil-seeds123-posthoc-response.md). This selected display is not an unselected three-seed stability validation.
<!-- /airfoil-seeds123-posthoc -->

<!-- darcy-pi-beta-calibration-20261117 -->
**Darcy PI β 校准结果，2026-11-17。** [完整数据、配置、图表与下载](data/rebuttal/2026-11/E4/pi_beta_calibration_20261117/README.md) · [当前E4入口](data/rebuttal/2026-11/E4/README.md)

共享β=1在85²/141²/211²上的测试mean rAUEC为0.998995/0.971548/0.969888。共32,700条校准记录和22,200条测试记录完成独立核验。85²仍接近Uniform、略差于原β0.5且仅1/3sampling seeds低于1；旧固定β结果与新校准版本分开标注。
<!-- /darcy-pi-beta-calibration-20261117 -->

<!-- darcy-resolution-analysis-20261117 -->
**Darcy 观测分辨率结果更新，2026-11-17。** [当前E4结果与完整分析](data/rebuttal/2026-11/E4/README.md) · [本次修改](data/rebuttal/2026-11/E4/UPDATE_20261117.md) · [分析图](figures/rebuttal/2026-11/E4/resolution_analysis.pdf)

同一85²训练模型在85²/141²/211²真实观测上的27,000次请求已重新核验。当前报告补齐固定比例/固定点数收益、coarse预算混杂和区域误差能量分析；原始指标保留。该组为一个training seed和三个sampling seeds，不替代其他Darcy训练cohort。
<!-- /darcy-resolution-analysis-20261117 -->

<!-- original-cdcl-correction-20261117 -->
**Original Airfoil Cd/Cl correction, 2026-11-17.** [Corrected results and protocol](data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/README.md) · [Current numerical summary](RESULTS_DETAIL.md) · [English reviewer reply](reviews/airfoil-original-cdcl-training-seeds-response.md) · [Run and verification status](read.me)

Transolver3 training seeds 0/1/2 are reevaluated with their matching CA heads, the original 110-case population and the original Cd/Cl-specific PI/CPG/CA policies. All **11,880 new case records** are complete. The separate original-checkpoint control reproduces **3,960/3,960** original case errors exactly. The earlier P0 Cd/Cl rows are retained **field-policy diagnostics**, superseded for comparison with original Table 1. The corrected package supplies every active three-training-seed Cd/Cl summary.
<!-- /original-cdcl-correction-20261117 -->

<!-- airfoil-p0-final-evidence -->
## Airfoil P0 final evidence

[P0 field-policy accuracy and timing; archived Cd/Cl diagnostics](data/rebuttal/2026-11/p0_airfoil_final_20261115/README.md) · [Reuse gate](data/rebuttal/2026-11/p0_airfoil_final_20261115/reuse_audit_final.json)

This controlled workflow adds Airfoil training seeds 1/2 and the CPG beta0 complete-output timing extension. Its linked result page distinguishes accepted data, observed running progress and remaining requirements; it updates after validation.
<!-- /airfoil-p0-final-evidence -->

## 当前结果与测试范围 / Current results

[本分支具体测试了什么](TESTED_CONTENTS.md) · [最新结果包](data/rebuttal/2026-11/current_results_20261115/README.md) · [模型与 head 清单](data/rebuttal/2026-11/current_results_20261115/model_inventory.csv)

各实验按数据集、训练 cohort、模型、采样方法、预算和计时边界分开标注；已完成、历史替代协议和仍未完成的内容均可在测试索引中查到。

<!-- completed-training-index -->
## Earlier verified follow-up snapshot

[Completed training inventory](data/rebuttal/2026-11/linearno_followup_20261114/completed_training/README.md) · [Accuracy, timing and figures](data/rebuttal/2026-11/linearno_followup_20261114/README.md) · [Score and density evidence](data/rebuttal/2026-11/linearno_followup_20261114/score_visualization/README.md)

The November 14 training inventory records 9 completed backbones and five matching trained heads; the current inventory is linked above.
It identifies the historical Darcy cohort separately and distinguishes completed training from pending evaluation.
The full three-training-seed follow-up remains partial.
<!-- /completed-training-index -->


## Standard benchmarks: three independent training seeds

This supplement publishes the completed Elasticity, Pipe and Darcy training-seed
supplement: **9 independently trained backbones, 9 matching CA heads and 66,600
test observations**. Results report mean and sample SD across three independent
training runs; inference-sampling variability is provided separately.

**三种子补充实验内容：Elasticity / Pipe / Darcy 的三独立训练种子补充实验。**
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
