# E5: Independent training seeds

Three independently initialized ShapeNet-Car T3 models each complete 400 epochs and a corresponding 200-epoch CA head. The historical mixed single-/two-GPU model is outside this cohort. The 19,200 native records, 22,200 metric rows and paired differences passed independent numerical checks.

Within each model, sampling-seed quantities are averaged before computing training SD over the three models. Field rAUEC is 0.588961±0.001733 (PI), 0.555017±0.005512 (CPG) and 0.555024±0.005512 (CA). Cd ordering changes with training seed: means are 0.633829±0.025238, 0.645406±0.033556 and 0.646050±0.033288. CPG/CA Field means nearly coincide; the Cd ranking is not stable across training runs.

## Files

- [training_seed_summary.csv](training_seed_summary.csv)
- [training_seed_rauec.csv](training_seed_rauec.csv)
- [sampling_variability.csv](sampling_variability.csv)
- [paired_training_differences.csv](paired_training_differences.csv)
- [checkpoint_registry.csv](checkpoint_registry.csv)
- [metrics_records.csv](metrics_records.csv)
- [paired_case_differences.csv](paired_case_differences.csv)
