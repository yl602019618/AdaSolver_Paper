# Nine-page main-text layout

The main text, including all four figures, both result tables, and the complete conclusion, ends on page 9. References start on page 10; appendices start on page 12. The compiled `paper.pdf` contains 37 pages in total.

## Changes

- Enable default `microtype` features for improved line breaking.
- Set figure placement to `[!htbp]` for Figures 1–4.
- Set `topfraction` and `bottomfraction` to 0.95, `textfraction` to 0.05, and `floatpagefraction` to 0.85.
- Allow three top floats, three bottom floats, and five total floats per text page.
- Set `textfloatsep` to `12pt plus 2pt minus 2pt`, `floatsep` to `10pt plus 2pt minus 2pt`, and `intextsep` to `8pt plus 2pt minus 2pt`.
- Condense the captions for Figures 3 and 4 from four lines to three. Figure 4 refers to Figure 3 for the shared density normalization.
- Add `\clearpage` before the bibliography.

Figure 1 remains on page 2. Both result tables and Figure 2 appear on page 8. Figures 3 and 4 and the complete conclusion appear on page 9.

All section text, equations, table contents, figure PDF assets, figure widths, bibliography entries, and conference style files are unchanged. No main-text material was moved into the appendix. Font sizes, baseline spacing, and page geometry are unchanged. Global microtypography and float settings also reflow references and appendices.

## Validation

Built with TeX Live 2025, pdfLaTeX, and latexmk. The final build has no undefined references, undefined citations, duplicate labels, or overfull boxes. The two underfull-vbox warnings also occur in the unmodified source build. The main pages were visually inspected, and the conclusion's final sentence is present on page 9.

Compile from the repository root:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

The existing `scripts/build_manuscript.py` workflow remains available to refresh `paper.pdf`.
