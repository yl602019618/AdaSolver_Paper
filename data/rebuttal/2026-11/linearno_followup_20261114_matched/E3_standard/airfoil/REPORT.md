# Airfoil LinearNO follow-up

Generated from local records: 2026-11-14T15:26:28.418767+00:00.

Shared task: pressure channel 3 on the 221×51 grid, 900 training and 100 validation geometries, fixed published 110-case positive-lift test population. Historical test visibility is disclosed in protocol.md.

| Model | Epochs | Training state | Accuracy accepted | Weight SHA-256 |
|---|---:|---|---|---|
| t3 | 1000 | completed | True | 02ebac3c09ca65122fe9617e348f0c34bfd20a7251ccbe657a67e86cdf58ac5c |
| linearno | 1000 | completed | True | 3cbaf227e2f3d0a3526c1e2923c475b42a70a1acf50920c4354f251d6b659d15 |
| linearno_conv | 1000 | completed | True | 7750dd330b76f129af21ed54235e1d1d79c3ceabebdda8446acb90579fa118d5 |

The main point models use native Transolver3 and the official LinearNO point-temp variant. The official conv-temp variant is reported separately at Full grid only. Its lack of variable-point support is recorded in model_compatibility.md.

Reconstruction uses accepted grid_qhull_sparse_apply_v1p1, with original integer-grid float64 coordinates and the original linear-plus-nearest rule. Every sampled prediction is reconstructed by both old and new implementations. Full predictions are saved once and explicitly aliased between reconstruction views.

Complete pressure-field area ratios below use each backbone’s own Uniform. Mean ± sample SD is across three sampling seeds of one training seed. CPG beta=0 is the separate mechanism control.

| Backbone | Method | Beta | rAUEC |
|---|---|---:|---:|
| Transolver3 | ca | 1.0 | 0.563249 ± 0.007347 |
| Transolver3 | cpg | 1.0 | 0.529345 ± 0.006744 |
| Transolver3 | cpg_beta0 | 0.0 | 7.901202 ± 0.267334 |
| Transolver3 | pi | 1.0 | 0.671098 ± 0.007489 |
| Transolver3 | uniform | N/A | 1.000000 ± 0.000000 |
| LinearNO_point_temp | ca | 1.0 | 0.589981 ± 0.025945 |
| LinearNO_point_temp | cpg | 1.0 | 0.559296 ± 0.021060 |
| LinearNO_point_temp | cpg_beta0 | 0.0 | 6.201754 ± 0.238420 |
| LinearNO_point_temp | pi | 1.0 | 0.680622 ± 0.030819 |
| LinearNO_point_temp | uniform | N/A | 1.000000 ± 0.000000 |

Full-grid pressure errors are directly comparable on the same 110 cases. The convolutional model is a separately trained Full-only supplement.

| Native model | Full relative L2 |
|---|---:|
| Transolver3 | 0.004092686 |
| LinearNO_point_temp | 0.004775013 |
| LinearNO_conv_temp_fullgrid | 0.002791744 |

Formal latency acceptance: False; observed groups 0/333, conditions 0/1323, formal repeats 0/132300. A is CPU input to float64 CPU field; B and C have separate rows.

Results: results/backbone_budget_metrics.csv, backbone_absolute_errors.csv, backbone_rauec.csv and backbone_latency_summary.csv. Missing files mean the corresponding stage is pending; no historical results are substituted. rAUEC is independently calculated per sampling seed, then reported as mean and sample SD across three sampling seeds; Full is excluded.

Commands and source identities: runtime/launches.json, runtime/*_workflow.json, config/execution_sources, config/reconstruction_sources, timing/sources. Raw predictions and selected indices: records/{architecture}; raw latency groups and occupancy records: timing/. Cross-task plots and archive assembly belong to the parent standard_benchmarks delivery.
