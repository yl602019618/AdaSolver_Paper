# AdaSolver paper

[Compiled paper](paper.pdf) · [LaTeX source](iclr2027_conference.tex)

Build with a TeX Live installation providing `latexmk`, `pdflatex`, and BibTeX:

```sh
python3 scripts/build_manuscript.py
```

The build updates `paper.pdf` and keeps temporary files in `.build/`. It preserves
the manuscript text, supplied figures, and result tables.

The `v1/` directory contains earlier paper material. Current editing starts from
`iclr2027_conference.tex` and its included sections.
