# Full-mesh-trained Airfoil pressure reference

This Transolver3 model is trained from scratch on pressure (Q channel 3), retaining all 11,271 points at every training step. Architecture, data preprocessing, optimizer, 1,000-epoch schedule and global batch of eight match the existing variable-point pressure backbone. Eight GPUs each process one geometry per global step; summed global-batch gradients are preserved. The checkpoint is selected by the same 200-case validation criterion used by the existing backbone. These benchmark cases are not a newly untouched holdout.

`training_manifest.json` and `training_curve.csv` document the completed training. `train_distributed.py` requires the AdaSolver package and accepts the dataset and output directories as arguments. `normalizer.npz` contains the pressure normalization statistics.

`records.csv` contains 1,100 actual evaluations: 110 cases with true Cl > 0.02, three sampling seeds at 37.5%, 50% and 75%, and one full-mesh evaluation per case. Lower-budget indices exactly match Uniform's indices. Predictions are reconstructed on the full reference mesh; the coefficient metric is absolute Cd/Cl error. `budget_curves.csv` contains the eight aggregated figure points, normalized by the shared variable-point backbone's full-mesh pressure-task error on the same cohort.

The older original Airfoil checkpoint predicts Mach (Q channel 4). Its old plotted values remain in the labeled historical data and are not used for this pressure comparison.
