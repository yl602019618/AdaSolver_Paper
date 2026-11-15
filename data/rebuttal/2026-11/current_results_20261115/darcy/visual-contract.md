# Darcy result visualization

These figures describe one matched training-seed0 cohort. Sampling-seed sampleSD
is distinct from across-training variability, which is not yet measured here.
All 200 historical smooth2 cases and all requested budget/method results remain
in the scalar data, including unfavorable CPG beta0 results.

Four figures answer separate questions: own-Uniform rAUEC; absolute error over
budget with Full-only convolution reference; error versus direct CPU complete
field latency; and old/new direct request and separate reconstruction costs.
Use common method colors and beta-specific markers, natural title case, and
PDF/SVG/300dpi PNG export. No diagram/image-generation model is used.

Accuracy uses 200 cases; timing uses three fixed cases, three sampling seeds,
20 warmups and100 formal repeats per condition. Full has no sampling seed.
Old/new cost ratios and plots never mix A/B/C boundaries. No profiled stage sum
is substituted for the direct clock. Figure-source CSVs are generated from the
public tables; raw fields and checkpoint files are not rendering inputs.
