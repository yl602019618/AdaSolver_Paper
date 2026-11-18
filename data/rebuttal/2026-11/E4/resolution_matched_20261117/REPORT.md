# Darcy resolution-matched training experiment

Status: all three accuracy matrices independently accepted; statistics and figure acceptance are recorded separately in results/statistics_acceptance.json and results/figures_receipt.json.

The three models are trained from scratch at 85², 141² and 211², using direct stride 5/3/2 nodes from the same 421² source cases. Each model uses smooth1 rows 0–899 for training and 900–999 for validation; smooth2 rows 0–199 are used for final evaluation only. The historical test population had been viewed in previous work and is not claimed to be a previously unseen holdout.

The new 85² model is an internal matched control. The author-facing main table retains the original paper 85² as a labelled historical reference and adds 141²/211². Its historical and new columns differ in training/selection, PI β and evaluation grid; controlled resolution conclusions use all three newly trained models.

Common-grid relative L2 is measured after reconstruction to 211². Native-grid values are stored separately. The original E4 linear-Delaunay plus nearest-fallback reconstruction, original flattened PI distance rule and original CPG grad_lap definition are preserved. Full native output is an identity path.

Every sampling seed first averages 200 case errors at each budget, integrates the 37.5/50/75% curve, and divides by its own model/grid/seed Uniform area. The table reports the mean and sample SD of these three ratios. These are three sampling seeds for training seed 0, not three trained models.

Regional statistics use physical_bands_0p02_direct_queries_v1_20261119: boundary distance is evaluated directly at each native coordinate, and interface distance is measured to the same common211 coefficient-edge nodes, with both widths fixed at 0.02. This offline classification correction removes 301 out-of-band boundary assignments at 141² from the original nearest-boolean mapping. Original regional records remain as legacy evidence; training, retained points, scores, target density, full-field errors and rAUEC are unchanged. Raw refinement scores, normalized display scores, pre-FPS planned allocation and actual retained indices are distinct stored quantities. PI target density is a nominal proposal, not an exact inclusion probability.

Accepted resolutions: 85², 141², 211².

Tables: [rebuttal main](results/tables/rebuttal_main.md), [controlled common](results/tables/controlled_common.md), [controlled native](results/tables/controlled_native.md).

Machine-readable statistics: results/budget_metrics.csv, absolute_errors.csv, rauec_by_seed.csv, rauec_summary.csv, regional_by_seed.csv and regional_summary.csv. Raw long-form records and retained prediction/index arrays are in records/r85, records/r141 and records/r211.

The experiment makes no formal latency claim. Training and head costs are kept in their manifests and curves.

## Result interpretation

在相同211²输出网格上，85²→141²→211²独立训练后的rAUEC，PI为0.9977 → 0.9904 → 1.0038；CPG为0.8398 → 0.9262 → 0.9569；CA为0.8400 → 0.9258 → 0.9571（sampling seeds=3，training seed=0）。

PI的边界保留率高于全局预算有0/9个条件，平滑区低于全局预算有9/9个条件；CPG的边界保留率高于全局预算有9/9个条件，平滑区低于全局预算有9/9个条件；CA的边界保留率高于全局预算有9/9个条件，平滑区低于全局预算有9/9个条件（条件为3个分辨率×3个预算，先按case和sampling seed聚合）。

Recompute: `python3 code/aggregate.py` then `python3 code/verify_statistics.py` and `python3 code/plot_results.py`. During execution, `aggregate.py --allow-partial` emits N/A for pending matched columns.
