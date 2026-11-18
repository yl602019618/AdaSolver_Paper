# Claim card

All three matched accuracy matrices are accepted.

在相同211²输出网格上，85²→141²→211²独立训练后的rAUEC，PI为0.9977 → 0.9904 → 1.0038；CPG为0.8398 → 0.9262 → 0.9569；CA为0.8400 → 0.9258 → 0.9571（sampling seeds=3，training seed=0）。

PI的边界保留率高于全局预算有0/9个条件，平滑区低于全局预算有9/9个条件；CPG的边界保留率高于全局预算有9/9个条件，平滑区低于全局预算有9/9个条件；CA的边界保留率高于全局预算有9/9个条件，平滑区低于全局预算有9/9个条件（条件为3个分辨率×3个预算，先按case和sampling seed聚合）。

Historical 85² is a reference, not a matched training cohort. The random-initialization interface checks are engineering validations only. Sampling SD does not estimate training variability.
