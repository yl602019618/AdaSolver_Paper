**car_training_seed_rauec.** Field rAUEC for three independently trained Transolver3 models. Colored points are training seeds0/1/2; black marks and error bars show the mean and sample SD across the three models. Each model value averages three matched sampling-seed area ratios over budgets37.5%,50%,75%. CPG and CA use beta1.

**car_training_seed_absolute_error.** Absolute field error, defined as velocity relative L2 plus0.5 times surface-pressure relative L2. Curves average cases and sampling seeds within each trained model, then show the mean and sample SD across three independent training seeds. The dotted line is the same three models evaluated at the full point set.

**car_backbone_rauec.** Each backbone uses its own same-checkpoint Uniform denominator. Symbols show the mean and sample SD across sampling seeds0/1/2 for training seed0. CPG beta0 is the paired compensation ablation; beta1 is the primary setting. Ratios do not rank absolute accuracy across backbones.

**car_backbone_absolute_error.** Absolute field errors for the two matched training-seed0 models on100 test cases. Error bars show sample SD across the three sampling-seed case means. Full is a same-checkpoint reference at100% budget, excluded from the rAUEC integral.

**car_absolute_error_latency.** Error uses100 test cases; latency is the directly measured median over3 fixed geometries,3 sampling seeds, and100 formal repeats after20 warmups. Full has one series per geometry. Both axes use the same backbone checkpoint and native reconstruction protocol. The boundary is CPU input to complete CPU float64 output with batch1 and eager FP32 networks. Points on each method curve correspond to37.5%,50%,75% budgets in order.

**reconstruction_fixed_prediction_parity.** Elementwise reconstruction agreement on108 fixed-prediction validation pairs at3 budgets and3 sampling seeds. Maximum absolute differences are shown per pair; all satisfy atol1e-6 plus rtol1e-5 times the original magnitude. These pairs validate the interpolation implementation on the recorded original checkpoints. Final new-checkpoint complete-request costs remain pending.

**car_training_seed_rauec_table.** ShapeNet-Car field rAUEC for Transolver3. Each training-seed entry averages the three matched sampling-seed ratios. Mean ± sample SD uses three independently trained models. CPG and CA use beta1; all ratios integrate37.5%,50%,75% budgets only.
