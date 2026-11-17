# LinearNO ShapeNet-Car PI/CA supplement

| Method | Field rAUEC ↓ | Cd rAUEC ↓ |
|---|---:|---:|
| Uniform | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| PI | 0.574891 ± 0.002469 | 0.596994 ± 0.025766 |
| CPG (β=1) | 0.531376 ± 0.002579 | 0.604585 ± 0.025192 |
| CA | 0.531345 ± 0.002640 | 0.603715 ± 0.024437 |

The ± values are sampling SD for the existing trained seed0 backbone. One new matching CA head was trained for200epochs; PI requires no head.

[Full package](data/rebuttal/2026-11/linearno_car_pi_ca_20261117/README.md) · [Verification](data/rebuttal/2026-11/linearno_car_pi_ca_20261117/verification.json) · [Inference costs](data/rebuttal/2026-11/linearno_car_pi_ca_20261117/inference_budget_summary.csv).
