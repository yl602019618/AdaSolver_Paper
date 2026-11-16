**Archived Cd/Cl panels.** Any Cd/Cl rows or panels in these P0 figures are force diagnostics from field-target policies. They are superseded for original Table1 comparisons by the [corrected original-Cd/Cl package](../../../../data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/README.md). The original figure and source values are preserved.

**training_seed_rauec.** Dots are individual training-model rAUECs after each model aggregates its three matched sampling-seed area ratios. Black marks show the mean; sample-SD bars are drawn only after three independent training models exist. Beta1 is primary and beta0 is the paired compensation ablation. All three independent training seeds are included.

**training_seed_error_budget.** Case means and sampling means are computed within each checkpoint before averaging independent training models. Error bars are the sample SD across three models when complete. The dotted Full reference uses the same independently trained checkpoints at100% points. All three independent training seeds are included.

**checkpoint_error_latency.** Each panel is one checkpoint and physical GPU. ScopeA latency averages formal condition medians over the fixed3-case/3-sampling-seed matrix;100 formal repetitions and20 warmups are retained per condition. Accuracy uses110 cases under the same checkpoint and reconstruction. No raw timings are pooled across training seeds or GPUs. All three independent training seeds are included.

**old_new_checkpoint_cost.** ScopeA at the fixed50% budget, separated by checkpoint/GPU. Crosses and circles compare original and rule-equivalent new reconstruction. All available methods, including beta0 and slower outcomes, remain visible. All three independent training seeds are included.

**checkpoint_cache_reuse.** ScopeC retains the measured first request and derives [first+(R−1)×steady median]/R for each case/sampling condition. Dashed CPU-map and solid GPU-map curves average these estimates only within one checkpoint. They are not contiguous R-request wall-time measurements. All three independent training seeds are included.

**training_seed_rauec_table.** rAUEC mean ± sampleSD across three independently trained models after within-model sampling aggregation. Pending SD is not substituted with zero. All three independent training seeds are included.
