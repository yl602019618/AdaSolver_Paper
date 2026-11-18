# Manuscript revisions

- Edit the clean manuscript in `iclr2027_conference.tex` and its included files.
- After every manuscript revision, run `python3 scripts/build_manuscript.py` to
  update both `paper.pdf` and `paper_marked.pdf`, together with the generated
  `iclr2027_conference_marked.tex` review source.
- Keep change-marking commands out of the clean sources. Do not edit the generated
  marked source by hand; regenerate it from the clean manuscript.
- The marked version shows cumulative changes against the Git revision in
  `docs/mark-baseline.txt`. Keep this baseline fixed unless the user asks to reset it.
- Check both builds for unresolved references and inspect the changed pages.
- When replacing graphics, retain the baseline assets in `review_assets/` with the
  marked TeX so its appended before/after comparison pages remain compilable.
