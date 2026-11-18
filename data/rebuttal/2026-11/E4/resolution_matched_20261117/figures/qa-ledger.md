# Render QA

| Issue | Artifact | Fix / evidence | Status |
|---|---|---|---|
| rAUEC axis offset could display Uniform as 0 plus an offset | A | Disable numeric offset; inspected final fixture PNG shows Uniform at1 | Fixed |
| Large score range hid lower-score structure | D | Labelled log(1+relative score) display; shared cross-resolution/method limits and no clipping | Fixed |
| Source score, planned density and actual retention could be confused | D | Separate labelled columns and colorbars; real selected indices plotted separately | Verified in fixture |
| Labels, legends, grids and exports | A/B/C/D | Six isolated, watermarked engineering groups exported PDF/SVG/300dpiPNG and inspected | Engineering check passed |
| Actual final scientific figures and E training/head curve | All 13 final groups | Actual A/B/C/E and all nine D plates inspected; three D plates expanded; all 39 file hashes, 300 dpi, PDF embedded fonts/bounds and SVG live text checked | Passed (2026-11-19) |

Engineering fixture outputs under evaluation/render_fixture are not experiment results.

Final visual receipt: `provenance/engineering/final_visual_review.json`. All nine D plates were checked in a contact sheet and expanded PI/CPG/CA plates cover cases 1000/1099/1199. Shared limits and all 27 retained-node counts were checked against the plotted CSVs. No plot source or numerical value was changed during QA. Heatmaps remain scientific raster layers within otherwise vector/text SVG/PDF exports.
