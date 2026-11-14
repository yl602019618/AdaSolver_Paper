Artifact: ShapeNet-Car reconstruction follow-up result figures and tables.
Destination: ICLR paper supplement/local reviewer package; quantitative figures.
Question: Under the same full-field rules, does implementation optimization change errors, reconstruction cost or complete request cost?
Source data: results/car_budget_metrics.csv, car_rauec.csv, car_latency_grouped.csv, reconstruction_summary.csv and their raw JSONL/CSV sources.
Evidence: paired correctness, sampling-seed uncertainty, and direct synchronized wall timing.
Statistics: rAUEC ratios are computed per seed, then mean and sample SD over3 seeds. Timing medians/p95 pool100 repetitions over3 fixed cases and3 sampling seeds (Full has no seed).
Panels: old/new reconstruction in separate rAUEC panels; absolute error with shared metric axes for both backbones; reconstruction/total cost by backbone with A/B/C kept in separate figures; absolute error versus scope-A latency.
Labels: Transolver3, LinearNO, Uniform, PI, CPG beta=0/1, CA, Full100%, retained budget(%), complete-field error, Field rAUEC, time(ms).
Palette: Okabe-Ito accents; Uniform gray, PI orange, CPG blue, CA green. Line styles/markers also encode implementation and backbone.
Sources: locked protocol and actual result CSVs only; no invented or legacy-filled numbers.
Exports: vector PDF, editable SVG text,300dpi PNG; deterministic plot_car.py and Markdown/LaTeX tables.
Availability: LinearNO Car PI/CA absent from retained existing E3 matrix; N/A with reason. Standard task figures are delivered by the parent owner.
Render QA: inspect actual PNG/PDF after formal results exist, including clipped labels/legend, common axes, units, color/grayscale distinctions and CSV traceability.
