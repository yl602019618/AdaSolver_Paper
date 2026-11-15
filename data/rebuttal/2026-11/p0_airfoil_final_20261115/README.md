# P0 Airfoil final results

Publication status: **VERIFIED_COMPLETE**. This package contains checked scalar results and reproducibility material for the two point backbones on the locked110-case Airfoil population. Statistical result acceptance and training progress are separate.

The already published seed0 point-model formal timing is complete: 1314/1314 legacy conditions are reused. The9 Full-only convolution conditions remain in the earlier supplementary release, outside this primary point-model matrix. Seed0 CPG beta0 is the new timing requirement: 324/324 additional conditions are currently accepted. Existing seed0 formal latency is not described as running or unavailable.

Current accepted primary counts:

| Item | Verified | Required for final |
|---|---:|---:|
| models | 6 | 6 |
| heads | 6 | 6 |
| training seeds | 3 | 3 |
| accuracy requests | 30360 | 30360 |
| physical conditions | 4914 | 4914 |
| formal repeats | 491400 | 491400 |
| warmups | 98280 | 98280 |
| memory conditions | 4914 | 4914 |
| parity pairs | 29700 | 29700 |
| full reference requests | 660 | 660 |

Completed training receipts currently show 6/6 backbones and 6/6 matching heads. These receipt counts do not fill missing accuracy, parity or timing cells. Training seeds complete against the expanded P0 accuracy-and-timing matrix are [0, 1, 2]; seed0's independently verified legacy scope remains valid.

Full uses no sampling seed and is outside rAUEC. Each checkpoint averages cases and sampling runs before across-training-seed summaries; sampling SD, training sampleSD and timing-repeat SD stay separate. The final target has4914 primary conditions:1638 for each training seed, including both CPG beta0 and beta1. The historical110-case positive-lift test population is not relabelled a new sealed holdout.

Latency boundaries A/B/C and recorded hardware remain explicit. Stage profiling does not substitute for direct request wall time; fixed-map reuse is a separately labelled scene. No missing metric is filled from another checkpoint, another head, another training seed or an unverified timing run.

See `verification.json` for scoped acceptance, `pending.json` for the remaining work, `RESULTS_DETAIL.md` when present for the numerical report, and `release_manifest.json` for exact published bytes. Weights, raw CFD fields, feature tensors and long private logs remain external. This snapshot does not stop running experiments.

From the repository root, recheck public numerical data with:

```sh
python3 reproducibility/rebuttal/2026-11/p0_airfoil_final_20261115/validate_results.py --data-dir data/rebuttal/2026-11/p0_airfoil_final_20261115 --output-dir "${TMPDIR:-/tmp}/p0-airfoil-validation" --require-complete
```

Accepted conditions require matching accuracy and memory evidence. Completed paired measurements awaiting memory are not yet included in the accepted counts above; an accepted beta0 count of zero does not assert that no beta0 measurement has run.

Observed execution progress is recorded separately in [execution_progress.json](execution_progress.json). These observations are not verified metric rows.

| Observed field | Value |
|---|---|
| controller_execution_status | RUNNING |
| models[0].training_seed | 0 |
| models[0].backbone | t3 |
| models[0].backbone_execution_status | completed |
| models[0].epochs_completed | 1000 |
| models[0].updates_completed | 113000 |
| models[0].epochs_requested | 1000 |
| models[0].recorded_physical_gpu | 2 |
| models[0].head_execution_status | completed |
| models[0].head_epochs_completed | 200 |
| models[0].accepted_accuracy_requests | 5060 |
| models[0].accepted_accuracy_metric_rows | 40480 |
| models[0].backbone_manifest | linearno_followup_20261114/standard_benchmarks/airfoil/checkpoints/t3/manifest.json |
| models[0].accuracy_receipt | linearno_followup_20261114/standard_benchmarks/airfoil/results/t3_accuracy_acceptance.json |
| models[1].training_seed | 0 |
| models[1].backbone | linearno |
| models[1].backbone_execution_status | completed |
| models[1].epochs_completed | 1000 |
| models[1].updates_completed | 113000 |
| models[1].epochs_requested | 1000 |
| models[1].recorded_physical_gpu | 3 |
| models[1].head_execution_status | completed |
| models[1].head_epochs_completed | 200 |
| models[1].accepted_accuracy_requests | 5060 |
| models[1].accepted_accuracy_metric_rows | 40480 |
| models[1].backbone_manifest | linearno_followup_20261114/standard_benchmarks/airfoil/checkpoints/linearno/manifest.json |
| models[1].accuracy_receipt | linearno_followup_20261114/standard_benchmarks/airfoil/results/linearno_accuracy_acceptance.json |
| models[2].training_seed | 1 |
| models[2].backbone | t3 |
| models[2].backbone_execution_status | completed |
| models[2].epochs_completed | 1000 |
| models[2].updates_completed | 113000 |
| models[2].epochs_requested | 1000 |
| models[2].recorded_physical_gpu | 0 |
| models[2].head_execution_status | completed |
| models[2].head_epochs_completed | 200 |
| models[2].accepted_accuracy_requests | 5060 |
| models[2].accepted_accuracy_metric_rows | 40480 |
| models[2].backbone_manifest | linearno_followup_20261114/reuse_review/seed_extension/runs/airfoil/seed1/checkpoints/t3/manifest.json |
| models[2].accuracy_receipt | linearno_followup_20261114/reuse_review/seed_extension/runs/airfoil/seed1/results/t3_accuracy_acceptance.json |
| models[3].training_seed | 1 |
| models[3].backbone | linearno |
| models[3].backbone_execution_status | completed |
| models[3].epochs_completed | 1000 |
| models[3].updates_completed | 113000 |
| models[3].epochs_requested | 1000 |
| models[3].recorded_physical_gpu | 1 |
| models[3].head_execution_status | completed |

Figure PDFs:

- [checkpoint_cache_reuse.pdf](../../../../figures/rebuttal/2026-11/p0_airfoil_final_20261115/checkpoint_cache_reuse.pdf)
- [checkpoint_error_latency.pdf](../../../../figures/rebuttal/2026-11/p0_airfoil_final_20261115/checkpoint_error_latency.pdf)
- [old_new_checkpoint_cost.pdf](../../../../figures/rebuttal/2026-11/p0_airfoil_final_20261115/old_new_checkpoint_cost.pdf)
- [training_seed_error_budget.pdf](../../../../figures/rebuttal/2026-11/p0_airfoil_final_20261115/training_seed_error_budget.pdf)
- [training_seed_rauec.pdf](../../../../figures/rebuttal/2026-11/p0_airfoil_final_20261115/training_seed_rauec.pdf)
- [training_seed_rauec_table.pdf](../../../../figures/rebuttal/2026-11/p0_airfoil_final_20261115/training_seed_rauec_table.pdf)

Redraw figures from public CSVs:

```sh
python3 reproducibility/rebuttal/2026-11/p0_airfoil_final_20261115/plot_final.py --data-dir data/rebuttal/2026-11/p0_airfoil_final_20261115 --output-dir figures/rebuttal/2026-11/p0_airfoil_final_20261115
```

Analysis code commit: `d6847602ea8fed642b49c399b53ceaa38a9af936`.
