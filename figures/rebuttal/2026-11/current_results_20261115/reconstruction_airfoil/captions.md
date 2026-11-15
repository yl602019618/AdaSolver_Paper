**field_rauec_seed0.** Each model uses training seed0. Mean and sample SD are across sampling seeds0/1/2, after case averaging and separate matched-Uniform area ratios over37.5%,50%,75%. Each backbone has its own denominator. Conv-temp is Full-only and has no rAUEC. Car LinearNO PI/CA are unavailable in this protocol.

**absolute_error_budget.** Complete field errors use100 Car cases or110 Airfoil cases. Error bars are sample SD across the three sampling-seed case means, with one independently trained checkpoint per panel. Dotted lines are same-checkpoint Full100% references. Car uses velocity relative L2 plus0.5 pressure relative L2; Airfoil uses pressure relative L2.

**absolute_error_direct_latency.** ScopeA uses directly measured CPU-input-to-complete-CPU-float64 latency, batch1 and eager FP32. Latency pools100 formal repetitions after20 warmups over3 fixed cases and3 sampling seeds; Full has no sampling seed. Accuracy uses the full100/110-case population. Each curve binds the same checkpoint and reconstruction; no old-checkpoint errors are paired with new times.

**old_new_complete_request_cost.** Direct scopeA request medians at the fixed50% budget, retaining every available primary method. Crosses are original reconstruction and circles are the new rule-equivalent implementation. Slower Airfoil requests are retained; sparse GPU application alone does not establish an end-to-end gain.

**fixed_map_cache_reuse_cost.** ScopeC at50% budget. Dashed lines use cached CPU application; solid lines use cached GPU application. Each point averages nine case/sampling-seed estimates of [measured first request+(R−1)×steady median]/R. These are derived reuse estimates, not contiguous R-request measurements. Coarse prediction, score, selection and network inference are recomputed; only the interpolation map is cached.

**field_rauec_table.** Field rAUEC mean ± sampling-seed sample SD for training seed0. Three matched sampling-seed ratios per model; each denominator is that backbone/checkpoint/population/reconstruction Uniform area. Car LinearNO PI/CA were not measured in this protocol.

**full_reference_table.** Same-checkpoint Full100% error and CPU-to-complete-CPU-field direct latency. Each model is training seed0; error uses100 Car or110 Airfoil cases and latency uses3 fixed cases with100 formal repetitions after20 warmups. The official Airfoil conv-temp variant is Full-only. Full has no sampling seed.
