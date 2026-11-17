# ShapeNet-Car LinearNO: PI and CA completion

Status: **VERIFIED_COMPLETE** for the scope below. This supplement adds the missing PI/CA training and inference path to the existing official ShapeNet-Car LinearNO training-seed 0 checkpoint.

The backbone is the complete official eight-block LinearNO, with hidden width 256, eight heads, seven native input channels, four output channels and 3,852,420 parameters. Its existing 400-epoch/142,000-update checkpoint and normalizer are reused. PI is geometry-only and requires no fitted head. CA uses a new 256–64–1 scalar pressure head trained on this LinearNO's own first-block features at exactly 10% points, with 710 training cases, 39 validation cases, two masks per case and 200 epochs. It never loads a Transolver head or reuses a Transolver slice cache. Final CA prediction executes the complete official LinearNO with point-key compensation.

All 100 original fold0 test cases, three point budgets (37.5%, 50%, 75%) and sampling seeds 0/1/2 are evaluated. Field and Cd use their separate frozen metric-specific policies. Field error is velocity relative L2 plus 0.5 times surface-pressure relative L2; Cd uses relative absolute total-drag error. In each sampling run, case means are integrated across the three budgets and normalized by the matching Uniform area from the same checkpoint. Full 100% is a separate reference and is not integrated.

| Method | Field rAUEC ↓ | Cd rAUEC ↓ |
|---|---:|---:|
| Uniform | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| PI | 0.574891 ± 0.002469 | 0.596994 ± 0.025766 |
| CPG (β=1) | 0.531376 ± 0.002579 | 0.604585 ± 0.025192 |
| CA | 0.531345 ± 0.002640 | 0.603715 ± 0.024437 |

The ± values are **sampling-seed sample SDs** for one trained backbone. No independent-training-seed SD is claimed. All 4,600 Full/Uniform/CPG control records reproduce the original field/Cd metric values and sampling-index identities exactly; new PI/CA policies add 3,600 rows, giving 8,200 native requests and 9,200 primary metric views. Cross-target diagnostic metrics remain in the raw records but do not replace metric-specific Cd or Field results. CPG β0/1 accuracy controls, including its small field degradation at β1, are retained separately.

The pressure head's 1,498 real coarse mask groups and 550,561 stored surface rows passed acceptance. Every target and selected index was checked, its saved validation MSE was reproduced, and its first-block feature output matched the official model's hook exactly on a real geometry.

Complete-output timings at 37.5% (plus the 100% Full reference) are:

| Policy target | Method | Retained points | Complete CPU-field request / ms | Full/request latency ratio |
|---|---|---:|---:|---:|
| Field | Full | 100% | 44.641 | 1.000 |
| Field | UNIFORM | 37.5% | 67.595 | 0.660 |
| Field | PI | 37.5% | 99.145 | 0.450 |
| Field | CPG | 37.5% | 195.488 | 0.228 |
| Field | CA | 37.5% | 191.963 | 0.233 |
| Cd | Full | 100% | 44.696 | 1.000 |
| Cd | UNIFORM | 37.5% | 67.614 | 0.661 |
| Cd | PI | 37.5% | 97.838 | 0.457 |
| Cd | CPG | 37.5% | 214.473 | 0.208 |
| Cd | CA | 37.5% | 210.527 | 0.212 |

PI and CA have higher complete-request latency than the same-GPU Full reference in all 12 target–budget comparisons.

The timed call starts from CPU-resident normalized inputs and a mesh-derived pressure-influence vector and returns a complete physical float64 CPU field. Method-specific geometry/scorer setup, dynamic acquisition, selection, compensation, input transfer, native final forward, decoding, k4 reconstruction and output transfer are included. Model/data loading, mesh-influence preparation and offline force/error evaluation are outside this request. Sampling and predictions are recomputed for every call; no field prediction or selected-set cache supplies the timed output.

Timing covers three fixed E3 geometries, three sampling seeds and all three budgets. Each condition has 20 warmups and 100 formal measurements. Field and Cd policies were measured on separate exclusive H20 GPUs, 4 and 5, respectively, with their own Full/Uniform/CPG references. The table reports means of nine case/sampling condition medians (three case medians for Full), not a pooled percentile. All 222 conditions, 22,200 formal repetitions and 4,440 warmups are included. Before each timing condition, output metrics and exact selected/coarse indices are checked against native accuracy records. Per-condition p95 and every repetition are available below. No isolated per-method memory result is claimed.

| Artifact | Contents |
|---|---|
| [per_case_records.csv.gz](per_case_records.csv.gz) | All 8,200 native requests, both measured metrics and policy/checkpoint/point identities |
| [primary_metric_records.csv.gz](primary_metric_records.csv.gz) | 9,200 explicitly assigned metric-specific views |
| [accuracy_summary.csv](accuracy_summary.csv) | Per-sampling-seed rAUEC, mean and sampling SD |
| [per_sampling_seed_budget.csv](per_sampling_seed_budget.csv) | Case-mean curves, method/Uniform areas and rAUEC |
| [uniform_denominators.csv](uniform_denominators.csv) | Six target/sampling-seed denominators |
| [frozen_policies.json](frozen_policies.json) | Frozen Field/Cd configurations |
| [checkpoint_registry.csv](checkpoint_registry.csv) | Backbone/head identities and external backbone locations |
| [head_inference_manifest.json](head_inference_manifest.json) | Portable pointer to the included trained head weights |
| [head_training_history.csv](head_training_history.csv) | Complete 200-epoch head history |
| [inference_budget_summary.csv](inference_budget_summary.csv) | All budgets and explicit same-GPU comparisons |
| [inference_conditions.csv](inference_conditions.csv) | 222 condition median/p95/mean/repeat-SD measurements |
| [inference_repeats.csv.gz](inference_repeats.csv.gz) | Every formal and warm-up wall-clock observation |
| [verification.json](verification.json) | Head, control, coverage, policy and independent numeric checks |
| [reproduction_command.md](reproduction_command.md) | Exact workspace execution and public CSV-only verification |

This completion is for training seed 0. Independent LinearNO Car backbone seeds 1/2, B/C reconstruction scenarios and isolated per-method memory remain outside this package. The official model implementation is referenced at its pinned upstream commit and is not bundled; the authored adapters, training/evaluation wrappers, inference callable and verification code are provided.
