# Nine-page main-text layout

The main text, including all four figures, both result tables, and the complete conclusion, ends on page 9. References start on page 10; appendices start on page 12. The compiled `paper.pdf` contains 39 pages in total. The new measure-correction ablation occupies Appendix D.1 on pages 26–27, before More Results (D.2) on page 28.

## Changes

- Enable default `microtype` features for improved line breaking.
- Set figure placement to `[!htbp]` for Figures 1–4.
- Set `topfraction` and `bottomfraction` to 0.95, `textfraction` to 0.05, and `floatpagefraction` to 0.85.
- Allow three top floats, three bottom floats, and five total floats per text page.
- Set `textfloatsep` to `9pt plus 1pt minus 2pt`, `floatsep` to `7pt plus 1pt minus 2pt`, and `intextsep` to `6pt plus 1pt minus 2pt`.
- Set the space above captions to 5 pt.
- Condense the captions for Figures 3 and 4 from four lines to three. Figure 4 refers to Figure 3 for the shared density normalization.
- Add `\clearpage` before the bibliography.
- Keep the Airfoil figure before the SuperWing paragraph with a float barrier, so the latter does not begin as a fragment at the bottom of page 8.
- Export the first slide of the updated author presentation as the cropped vector Figure 1 and its 600 dpi PNG.

Figure 1 remains on page 2. Both result tables and Figure 2 appear on page 8. Figures 3 and 4 and the complete conclusion appear on page 9.

No main-text material was moved into the appendix. Font sizes, baseline spacing, page geometry, figure widths, and conference style files are unchanged. The Airfoil paragraph states its 18.75% budget explicitly. The SuperWing description matches the existing Figure 4 records: a 37.5% budget and drag minimization at a fixed-lift target. Appendix D.1 adds the completed beta sweep, a comparison table, and its mechanism analysis.

## Validation

Built with pdfLaTeX and latexmk. The final build has no undefined references, undefined citations, duplicate labels, or overfull boxes. All PDF fonts are embedded. The first nine pages and the two-page D.1 section were visually inspected; the conclusion's final sentence is present on page 9.

Compile from the repository root:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

The existing `scripts/build_manuscript.py` workflow remains available to refresh `paper.pdf`.
