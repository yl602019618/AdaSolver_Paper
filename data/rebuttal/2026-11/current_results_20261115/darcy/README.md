# Darcy matched 900/100 results — audited 15 November 2026

This package contains completed training-seed0 experiments, independently replayed from the original scalar records. It adds no training, GPU inference or modifications to the source experiment directory. The three model variants are Transolver3, official LinearNO point-temp, and the native convolution-temp Full-only supplement. They are not three independent training seeds.

All three backbones used900 training and100 validation cases, common900-only normalizers,1000epochs and225000optimizer updates. The two adaptive backbones each have their own1000-epoch CA head bound to that backbone's hash. The convolution model has no adaptive/head task because its native input is fixed85x85. Backbone/head selection used validation data; historical visibility of the smooth2 test population remains disclosed.

The accuracy test covers all200 smooth2 cases. Each point backbone evaluates Full once per case plus Uniform, PI, CPG beta0, CPG beta1 and CA at37.5%,50%,75% with sampling seeds0/1/2:9200 physical requests each. Conv contributes200Full requests. Old/new reconstruction views produce37200 scalar metric rows. The3600 CPG beta pairs preserve coarse indices, coarse predictions and final selected indices; only compensation changes. All unfavorable beta0 results are retained.

For each model/reconstruction and sampling seed, rAUEC integrates the case-mean error curve at the three budgets and divides by that model's matched Uniform area. Full is excluded. The reported SD is across three sampling seeds. Across-training-seed SD remains pending; independently normalized rAUEC is not an absolute cross-model accuracy ranking.

Formal timing covers333 paired groups and1323 conditions. Each condition has20warmups and100 formal direct wall-time measurements with GPU boundary synchronization. The three scenes are A CPU model-ready input to complete CPU float64 field; B GPU-resident input to complete GPU float64 field; and C explicitly reused geometry/selected-set reconstruction maps while acquisition and predictions are recomputed. Model/kernel startup and first-request costs are separate. Full has one physical series per model/case/scene, without fabricated sampling repeats or duplicate old/new timings. All timing ran under the recorded exclusive H20 GPU0 admission. Independent memory comprises11 model/method processes and1323 conditions.

`latency_raw_repeats.csv.gz` stores the original compact warmup/formal observations; `backbone_latency_summary.csv` maps each group/condition to its checkpoint, head, population, boundary and code hashes. Every median,p95,mean and sampleSD was recomputed. `latency_profiles.csv.gz` contains separate instrumented passes, not an additive substitute for direct latency. Reuse-cost estimates retain their explicit derived-estimate definitions. Memory covers PyTorch allocator peaks and post-request NVML samples, not CPU SciPy allocations or all transient external-library GPU peaks.

The original controller's generic final_accuracy flag did not read Darcy's actual acceptance keys. This audit checks `model_specific_validation_accepted`, complete raw matrices and exact latency/memory coverage, and supplies the previously pending independent scalar aggregation. It does not silently change the original receipt strings.

Weights, full physical fields, caches and long logs remain external. The source index records immutable hashes. Public CSVs contain scalar results, identities and training histories. Historical training Git IDs are not manufactured from the paper baseline. The official LinearNO source pin is3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269.

Full-field relative-L2 errors:

| Model | Full error |
|---|---:|
| Transolver3 | 0.018292128 |
| LinearNO_point_temp | 0.008367571 |
| LinearNO_conv_temp | 0.005691229 |

New-reconstruction rAUEC (mean ± sampling sampleSD):

| Model | Method | rAUEC | Sampling SD |
|---|---|---:|---:|
| LinearNO_point_temp | uniform | 1.000000 | 0.000000 |
| LinearNO_point_temp | pi | 0.993120 | 0.023507 |
| LinearNO_point_temp | cpg_beta0 | 4.191984 | 0.007686 |
| LinearNO_point_temp | cpg | 0.710083 | 0.001465 |
| LinearNO_point_temp | ca | 0.708281 | 0.001003 |
| Transolver3 | uniform | 1.000000 | 0.000000 |
| Transolver3 | pi | 0.998867 | 0.018456 |
| Transolver3 | cpg_beta0 | 3.064428 | 0.036718 |
| Transolver3 | cpg | 0.844065 | 0.004417 |
| Transolver3 | ca | 0.842786 | 0.004271 |

Completed versus pending tests are enumerated in `test_scope.csv`; detailed evidence and limits are in `source_audit.json`. No three-training-seed claim is made for this seed0 package.


Figures (PDF, SVG and 300dpi PNG):

- `figures/darcy_rauec`: own-Uniform normalization, including the unfavorable beta0 ablation.
- `figures/darcy_absolute_error_budget`: absolute field error and three native Full references.
- `figures/darcy_error_latency`: absolute error versus directly measured scene-A complete output.
- `figures/darcy_reconstruction_cost`: matched old/new direct request cost and separate reconstruction profiling in A/B/C.

Each plot has a source CSV and caption. Redraw and validate the public package without model weights or raw CFD data:

```sh
python3 plot_results.py --data-dir . --output-dir figures
python3 validate_public.py --data-dir .
```

Raw timing repetitions and profiling scalars use gzip-compressed CSV, with unchanged numeric strings. Original long process logs, weights and prediction arrays remain external; their source hashes are indexed.

Model-specific reconstruction QA adds five accepted receipts with273 validation rows, zero violating elements and all downstream metric tolerances satisfied. Separate model/kernel startup and resident-input transfer records are in `preparation_setup.json` and `input_preparation.json`; exact reconstruction and training contracts are in `evaluation_protocol.json` and `training_protocol.json`.

Published reproduction scripts are under `reproducibility/rebuttal/2026-11/current_results_20261115/darcy`. Commands using script filenames refer to that code directory; the data directory is this package. Source manifests and original audit hashes are provenance snapshots; the current branch-wide byte hashes are in ../release_manifest.json.
