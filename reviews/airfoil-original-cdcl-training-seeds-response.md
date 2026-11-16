# Response on sampling and training seeds

The three seeds underlying the original Table 1 error bars were inference sampling seeds for one frozen Transolver3 checkpoint. We have now added evaluations of three independently trained Transolver3 models, using training seeds 0, 1 and 2 from the P0 study and the CA head associated with each model.

We evaluate the original 110-case Airfoil population with reference pressure-force Cl > 0.02, retaining the original exclusion of cases 1041 and 1139. We use the original Cd/Cl-specific PI, CPG and CA policies, budgets 37.5%, 50% and 75%, and sampling seeds 0, 1 and 2. Reconstruction and pressure-force integration follow the original implementation. As a protocol control, fresh evaluation of the original frozen checkpoint reproduced all 3,960 published per-case Cd/Cl errors exactly.

For each training checkpoint and sampling seed, we average errors over cases at each budget, integrate the three-budget curve, and divide by the matching Uniform area. We average the resulting three sampling-seed rAUECs within each checkpoint, then compute the mean and sample SD across the three independently trained models:

| Method | Training seed 0 | Training seed 1 | Training seed 2 | Mean ± training sample SD |
|---|---:|---:|---:|---:|
| PI | 2.110458 | 0.310785 | 0.716833 | 1.046026 ± 0.943918 |
| CPG | 3.383230 | 0.303888 | 0.706133 | 1.464417 ± 1.673868 |
| CA | 2.041773 | 0.326603 | 0.731440 | 1.033272 ± 0.896537 |

The last column is **training SD**, computed from three checkpoint-level means. Sampling SD describes the three inference-sampling runs within one checkpoint and is reported separately in the corrected package. Full 100% does not enter the rAUEC integral.

Airfoil Cd/Cl shows substantial variation because small predicted lift can amplify ratio errors. The retained cases 1007, 1059 and 1198 have reference Cl values 0.0241764, 0.0246764 and 0.0244129. For case 1198, training seed0 with CPG at 37.5% and sampling seed1 predicts Cl=0.0003855, giving a Cd/Cl absolute error of 51.2842. Uniform for training seed1 at the same budget and sampling seed predicts Cl=0.0002175, giving error 88.0548 and increasing that run's normalization area. All such observations remain in the results. The three-model mean rAUECs are above the Uniform reference of 1, and the per-model results vary widely.

The original frozen model is retained as a separate protocol control. The new three-model statistics use the P0 models' common 900/100 training/validation split and fitted normalizer. The historical control uses its original training setup and normalizer. We report these cohorts separately and make no paired improvement claim between the original control and the new seed0. This supplement covers Airfoil. It makes no three-training-seed completion claim for Plasticity.

The response and current Cd/Cl summary use the [corrected original-Cd/Cl package](https://github.com/yl602019618/AdaSolver_Paper/tree/rebuttal/20261114-linearno-seeds/data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117), which contains every case record, all nine paired Uniform denominators, checkpoint/head bindings and separate sampling/training summaries.
