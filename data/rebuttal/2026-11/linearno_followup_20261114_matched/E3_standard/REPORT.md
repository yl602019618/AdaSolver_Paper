# LinearNO standard benchmark follow-up

Generated from current receipts at 2026-11-14T15:24:00.218449+00:00.

The main comparison uses Transolver3 and the official LinearNO point-temp model on Darcy 85×85 and Airfoil 221×51. The official convolution variant is a separate Full-only reference. Both point models share a new 900/100 training/validation split, training seed 0, and the same task-specific point schedule. Historical test visibility is disclosed in protocol.md.

| Dataset | Backbone | Accuracy stage | Field rows | Reason if pending |
| --- | --- | --- | ---: | --- |
| darcy | Transolver3 | pending | N/A | Training or complete preliminary evaluation pending |
| darcy | LinearNO_point_temp | pending | N/A | Training or complete preliminary evaluation pending |
| darcy | LinearNO_conv_temp | pending | N/A | Training or complete preliminary evaluation pending |
| airfoil | Transolver3 | complete | 10120 |  |
| airfoil | LinearNO_point_temp | complete | 10120 |  |
| airfoil | LinearNO_conv_temp | complete | 220 |  |

Complete accuracy matrix: False. Complete A/B/C latency matrix: False.

Current independently checked field rows: 20460; rAUEC rows: 20; formal timing conditions: 0 / 2,646.

The independent collector verifies the predeclared test IDs, checkpoint and normalizer, exact method/budget/seed matrix, and CPG beta pairing. rAUEC integrates each seed’s case-average error curve at 37.5%, 50%, and 75%, divides by its own matched Uniform area, then reports mean and sample SD. Full 100% does not enter the integral. Each backbone has its own denominator.

Final predictions use reconstruction v1.1: original QHull linear interpolation with nearest fallback, and a sparse application of the same weights. Both reconstructions consume the same fixed predictions. Each new checkpoint must pass validation and full-test equivalence checks.

Tables and figures are generated from results/*.csv by code/render_results.py. The render manifest records source CSVs; visual QA is separate. Missing cohorts or CA rows are N/A, and incomplete timing is not replaced with a single evaluation-call duration. Full-only convolution results are kept out of rAUEC.

Primary timing A is CPU input to a complete float64 CPU field. B is GPU-resident input to a complete float64 GPU field. C reuses only an interpolation map for identical geometry and retained point order. Formal results contain 20 warmups and 100 direct wall-time repeats per condition; reconstruction profiling and memory runs are separate.

Training and raw arrays remain indexed in the task directories. The local archive excludes checkpoint payloads and large training caches. Experiment scripts perform no Git writes. The user separately authorized root to publish the reviewed current snapshot to an independent branch; see ../coordination/publication.json. Main and Feishu remain unchanged.
