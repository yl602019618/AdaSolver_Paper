# Three independent training seeds on standard benchmarks

Elasticity, Pipe and Darcy each use three independently initialized Transolver3 backbones and a separately trained CA head for each backbone. Every model is evaluated on the same 200 test cases, with three inference sampling seeds at each of the 37.5%, 50% and 75% budgets. Full-mesh inference uses the same checkpoint at 100% of its reference points.

The tables report mean ± sample standard deviation across the three training runs. Within each training run, errors first average over test cases and sampling seeds. Sampling variability is recorded separately.

## Relative area under the error curve

| Dataset | Uniform | PI | CPG | CA |
| --- | ---: | ---: | ---: | ---: |
| Elasticity | 1.000000 ± 0.000000 | 0.726406 ± 0.002168 | 0.467020 ± 0.001175 | 0.560134 ± 0.010647 |
| Pipe | 1.000000 ± 0.000000 | 0.855711 ± 0.012698 | 0.654441 ± 0.010113 | 0.692265 ± 0.009394 |
| Darcy | 1.000000 ± 0.000000 | 0.995973 ± 0.002841 | 0.639290 ± 0.015871 | 0.639412 ± 0.016405 |

For each sampling seed, rAUEC divides the method's trapezoidal error area by the matched Uniform area over budgets [0.375, 0.5, 0.75]. These three ratios are averaged within each training run before computing the training-run mean and SD. This follows the companion ShapeNet-Car E5 analysis.

## Individual training runs

| Dataset | Method | Training seed 0 | Training seed 1 | Training seed 2 | Mean ± training SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Elasticity | UNIFORM | 1.000000 | 1.000000 | 1.000000 | 1.000000 ± 0.000000 |
| Elasticity | PI | 0.724283 | 0.728616 | 0.726319 | 0.726406 ± 0.002168 |
| Elasticity | CPG | 0.465702 | 0.467959 | 0.467399 | 0.467020 ± 0.001175 |
| Elasticity | CA | 0.559724 | 0.570980 | 0.549699 | 0.560134 ± 0.010647 |
| Pipe | UNIFORM | 1.000000 | 1.000000 | 1.000000 | 1.000000 ± 0.000000 |
| Pipe | PI | 0.870322 | 0.847347 | 0.849465 | 0.855711 ± 0.012698 |
| Pipe | CPG | 0.661021 | 0.642795 | 0.659506 | 0.654441 ± 0.010113 |
| Pipe | CA | 0.698338 | 0.681445 | 0.697012 | 0.692265 ± 0.009394 |
| Darcy | UNIFORM | 1.000000 | 1.000000 | 1.000000 | 1.000000 ± 0.000000 |
| Darcy | PI | 0.997769 | 0.997454 | 0.992697 | 0.995973 ± 0.002841 |
| Darcy | CPG | 0.635426 | 0.656736 | 0.625707 | 0.639290 ± 0.015871 |
| Darcy | CA | 0.635381 | 0.657456 | 0.625398 | 0.639412 ± 0.016405 |

## Full-field relative L2 error

| Dataset | Method | 37.5% | 50% | 75% | Full mesh (100%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Elasticity | Full | — | — | — | 0.008786 ± 0.000617 |
| Elasticity | UNIFORM | 0.120960 ± 0.000170 | 0.085413 ± 0.000271 | 0.045471 ± 0.000088 | — |
| Elasticity | PI | 0.084729 ± 0.000058 | 0.061956 ± 0.000062 | 0.034720 ± 0.000205 | — |
| Elasticity | CPG | 0.060689 ± 0.000254 | 0.040529 ± 0.000083 | 0.018176 ± 0.000267 | — |
| Elasticity | CA | 0.068452 ± 0.000490 | 0.048374 ± 0.000745 | 0.024323 ± 0.001143 | — |
| Pipe | Full | — | — | — | 0.004585 ± 0.000228 |
| Pipe | UNIFORM | 0.010793 ± 0.000211 | 0.008572 ± 0.000247 | 0.006318 ± 0.000238 | — |
| Pipe | PI | 0.009183 ± 0.000187 | 0.007272 ± 0.000204 | 0.005526 ± 0.000266 | — |
| Pipe | CPG | 0.006028 ± 0.000187 | 0.005478 ± 0.000217 | 0.004852 ± 0.000231 | — |
| Pipe | CA | 0.006379 ± 0.000171 | 0.005816 ± 0.000233 | 0.005100 ± 0.000228 | — |
| Darcy | Full | — | — | — | 0.010172 ± 0.000301 |
| Darcy | UNIFORM | 0.021429 ± 0.000589 | 0.018054 ± 0.000463 | 0.013408 ± 0.000320 | — |
| Darcy | PI | 0.021653 ± 0.000580 | 0.017947 ± 0.000500 | 0.013250 ± 0.000332 | — |
| Darcy | CPG | 0.011328 ± 0.000318 | 0.011037 ± 0.000279 | 0.010509 ± 0.000280 | — |
| Darcy | CA | 0.011354 ± 0.000305 | 0.011046 ± 0.000289 | 0.010489 ± 0.000285 | — |

## Training and evaluation protocol

| Dataset | Full points | Backbone training points | Backbone epochs | CA head epochs | Train / validation / test cases |
| --- | ---: | --- | ---: | ---: | --- |
| Elasticity | 972 | 389–972 | 500 | 200 | 1000 / 200 / 200 |
| Pipe | 16641 | 6656–16641 | 1000 | 1000 | 1000 / 200 / 200 |
| Darcy | 7225 | 7225 | 1000 | 1000 | 1000 / 200 / 200 |

All backbones use eight Physics-Attention blocks, hidden width 128, eight attention heads and 64 slices. They train in FP32 with AdamW, a peak learning rate of 1e-3, weight decay 1e-5 and gradient clipping at 0.1. Elasticity uses cosine annealing; Pipe and Darcy use OneCycleLR. Checkpoints minimize physical-space validation relative L2. The test cases are excluded from checkpoint selection.

Elasticity uses training IDs 0–999, validation IDs 1000–1199 and test IDs 1600–1799. Pipe uses training IDs 0–999, validation IDs 1200–1399 and test IDs 1000–1199. Darcy trains on smooth1 rows 0–999, validates on smooth2 rows 200–399 and tests on smooth2 rows 0–199. Normalizers are fitted on training samples.

CA is trained separately for each backbone. Elasticity uses a 128→4→1 stress head on frozen first-block features from 48-, 72- and 97-point inputs; all three Elasticity heads use FP32 CPU feature extraction and training. Pipe and Darcy retain their complete two-block Transolver heads trained on GPU. Pipe head training resamples 1664–16641 points; Darcy uses 722-point inputs. Coarse inference uses 97, 1664 and 722 points for Elasticity, Pipe and Darcy, respectively.

Sampling parameters are fixed to their prior production settings. Pipe parameters were historically calibrated on the first 50 cases of its current test cohort, and Darcy parameters also have a test-informed development history. The new training-seed study retains these policies without retuning them. Its backbone and head checkpoint selection uses the separate validation cohorts above.

## Results by dataset

- Elasticity: CPG has the lowest mean rAUEC (0.467020 ± 0.001175) and the lowest rAUEC in all three training runs.
- Pipe: CPG has the lowest mean rAUEC (0.654441 ± 0.010113) and the lowest rAUEC in all three training runs.
- Darcy: CPG and CA have nearly identical three-run mean rAUEC (difference 0.000122). Their ordering varies across training runs: the lowest method in seeds 0, 1 and 2 is CA, CPG, CA.

## Files

- `training_seed_rauec.csv`: every training-run rAUEC and its three-run mean and SD.
- `training_seed_summary.csv`: absolute errors by method and final point budget.
- `sampling_variability.csv`: sampling-seed variation within each trained model.
- `paired_training_differences.csv`: matched method differences for each training run.
- `checkpoint_registry.csv`: the nine backbone/head pairs, selection epochs, training costs and identities.
- `metrics_records.csv.gz`: all 66,600 per-case observations.
- `verification.json` and `matrix_checks.csv`: matrix, checkpoint, point-set and numerical aggregation checks.

Checkpoints and full selected-point arrays remain in the experiment workspace. Registry paths are relative to that workspace; the Git supplement contains the numeric evidence and reproduction drivers.
