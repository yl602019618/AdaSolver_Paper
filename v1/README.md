# AdaTransolver ICLR 2027 draft

This directory contains a self-contained anonymous LaTeX draft.

## Build

Compile `main.tex` with BibTeX support. For example, with Tectonic:

```powershell
tectonic -X compile main.tex --keep-logs
```

The current environment uses the ICLR 2026 public style as the closest official
shell, with only the review header year changed to 2027. Replace
`iclr2026_conference.sty` and `iclr2026_conference.bst` with the official ICLR
2027 files when they are published, then update the two package/style references
in `main.tex`.

## Files

- `main.tex`: anonymous English manuscript draft.
- `references.bib`: BibTeX entries used by the draft.
- `AUTHOR_NOTES.md`: terminology ledger, claim--evidence map, and submission blockers.
- `figures/`: current experiment figures; re-export embedded Chinese labels in English before submission.
- `main.pdf`: rendered review copy produced from the current sources.

The author list, affiliations, final experiment registry, confidence intervals,
and equal-time comparisons remain author inputs before submission.
