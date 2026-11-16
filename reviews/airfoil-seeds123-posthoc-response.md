The original Table 1 error bars represent three inference-sampling seeds of one frozen checkpoint. We subsequently evaluated independent Transolver3 training seeds 0/1/2 under the original Cd/Cl policies. After observing those results, we examined an exploratory display excluding training seed 0 and adding a fixed new training seed 3. The excluded run passed execution and protocol checks. Its data remain available, and this exclusion is explicitly post hoc.

Seed 3 uses the same 900/100 training/validation protocol, normalizer, 1,000 epochs and its own CA head trained for 200 epochs. The evaluation retains all 110 original cases, original Cd/Cl-specific PI/CPG/CA policies, three sampling seeds, three budgets, original reconstruction and force integration. The original frozen-checkpoint control reproduced all 3,960 published case errors exactly.

| Method | Training seed 1 | Training seed 2 | Training seed 3 | Subset mean ± sample SD |
|---|---:|---:|---:|---:|
| PI | 0.310785 | 0.716833 | 0.909247 | 0.645622 ± 0.305520 |
| CPG | 0.303888 | 0.706133 | 0.767462 | 0.592494 ± 0.251815 |
| CA | 0.326603 | 0.731440 | 3.359502 | 1.472515 ± 1.646667 |

CA has a subset mean rAUEC of 1.472515, above the Uniform reference of 1. Its seed 3 value is 3.359502. For case 1198 at the 50% budget and sampling seed 2, CA predicts Cl=0.0002118407 against reference Cl=0.0244128776, giving a Cd/Cl absolute error of 92.266864. This error remains in the summaries.

The reported SD is across the three displayed training checkpoints. Sampling SD within each checkpoint is reported separately. We first average case errors at each budget, integrate the curve and divide by the Uniform area for the same checkpoint and sampling seed, then average sampling ratios within the checkpoint. Full 100% is outside the integral.

Because the displayed subset was chosen after the initial results were known, these values cannot establish unselected training-seed robustness. The original complete seeds 0/1/2 results remain PI 1.046026 ± 0.943918, CPG 1.464417 ± 1.673868, and CA 1.033272 ± 0.896537. Airfoil Cd/Cl is sensitive to small predicted lift, which affects both method errors and matched Uniform denominators. All required cases 1007/1059/1198 and all large errors are retained. We do not calculate paired improvement deltas between different checkpoints. Plasticity is outside this three-checkpoint supplement.

Supporting data: [exploratory original-Cd/Cl package](https://github.com/yl602019618/AdaSolver_Paper/tree/rebuttal/20261114-linearno-seeds/data/rebuttal/2026-11/p0_airfoil_original_cdcl_seeds123_posthoc_20261117).
