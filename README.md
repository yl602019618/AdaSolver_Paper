# AdaSolver paper

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
| Fig3: Airfoil design | [PNG](figures/fig3_airfoil_optimization.png) · [PDF](figures/fig3_airfoil_optimization.pdf) |
| Fig4: SuperWing design | [PNG](figures/fig4_superwing_optimization.png) · [PDF](figures/fig4_superwing_optimization.pdf) |
| Additional design comparisons | [Figures, PPT, panels, and results](supplementary/design_transfer/README.md) |
| Current inverse-design settings | [LaTeX](sections/inverse_design_settings.tex) · [Parameters](data/inverse_design_settings.json) |
