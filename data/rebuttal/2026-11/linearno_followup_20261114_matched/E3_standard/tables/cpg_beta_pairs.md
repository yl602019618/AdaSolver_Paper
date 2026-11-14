| Dataset | Backbone | Reconstruction | beta=0 | beta=1 | Paired delta (1−0) |
| --- | --- | --- | ---: | ---: | ---: |
| airfoil2d | Transolver3 | grid_scipy_linear_nearest_native_v1 | 7.9012 ± 0.2673 | 0.5293 ± 0.0067 | -7.3719 ± 0.2732 |
| airfoil2d | Transolver3 | grid_qhull_sparse_apply_v1p1 | 7.9012 ± 0.2673 | 0.5293 ± 0.0067 | -7.3719 ± 0.2732 |
| airfoil2d | LinearNO (point-temp) | grid_scipy_linear_nearest_native_v1 | 6.2018 ± 0.2384 | 0.5593 ± 0.0211 | -5.6425 ± 0.2193 |
| airfoil2d | LinearNO (point-temp) | grid_qhull_sparse_apply_v1p1 | 6.2018 ± 0.2384 | 0.5593 ± 0.0211 | -5.6425 ± 0.2193 |

Each paired comparison shares the coarse prediction and retained indices. Negative delta favors beta=1. SD is across three sampling seeds.
