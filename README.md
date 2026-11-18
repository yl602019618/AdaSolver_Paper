# AdaSolver paper

[Clean paper](paper.pdf) · [Clean LaTeX source](iclr2027_conference.tex) ·
[Marked paper](paper_marked.pdf) · [Marked LaTeX source](iclr2027_conference_marked.tex)

Build with TeX Live (`latexmk`, `pdflatex`, BibTeX, `latexdiff`, and `ulem`), Python 3.12+, and Poppler
(`pdfinfo`, `pdftotext`):

```sh
python3 scripts/build_manuscript.py
```

The build updates both `paper.pdf` (clean) and `paper_marked.pdf` (review), and
generates `iclr2027_conference_marked.tex`. Temporary files stay in `.build/`.
Edit the clean sources; the marked version is generated automatically. The build
uses the supplied figures and result tables without regenerating them.

The marked version compares the current clean manuscript with the Git commit in
[`docs/mark-baseline.txt`](docs/mark-baseline.txt), initially `d4021228`, the version
before the Section 3.1 revision. Added text is blue and underlined; deleted text
is red and struck through. Changed display equations are marked as whole blocks.
New and replaced graphics have blue frames in the marked manuscript. For images
replaced at the same path, the build saves the baseline assets in `review_assets/`
and appends unnumbered Baseline/Current comparison pages to the marked PDF.
Include `review_assets/` when sharing the marked TeX source.
Changed numeric cells in the ablation table show the old value above the new
value so both remain readable within the page width.
This fixed baseline keeps later edits visible cumulatively. Both PDFs must compile
successfully before the published pair is refreshed.

To compare against another commit for one build, use
`python3 scripts/build_manuscript.py --baseline <commit>`. To start a new review
round permanently, update `docs/mark-baseline.txt` to the accepted commit. Run the
default command again to return to the saved baseline after a one-off comparison.

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
