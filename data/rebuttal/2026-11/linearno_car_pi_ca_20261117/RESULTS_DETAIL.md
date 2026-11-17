# LinearNO ShapeNet-Car PI/CA results

One backbone training seed0; mean±sample SD across three inference sampling seeds. Separate frozen policies for Field and Cd.

| Method | Field rAUEC ↓ | Cd rAUEC ↓ |
|---|---:|---:|
| Uniform | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| PI | 0.574891 ± 0.002469 | 0.596994 ± 0.025766 |
| CPG (β=1) | 0.531376 ± 0.002579 | 0.604585 ± 0.025192 |
| CA | 0.531345 ± 0.002640 | 0.603715 ± 0.024437 |

Complete CPU-field timing at37.5% follows; all budgets are in inference_budget_summary.csv.

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

Across the 12 PI/CA target–budget summaries, 12 have higher complete-request latency than their same-GPU Full reference. These measurements do not turn reduced point counts into an unconditional end-to-end speedup claim.
