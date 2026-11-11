# Elasticity timing with 97 coarse points

These measurements use the frozen Elasticity accuracy configurations recorded
in `configurations.json`. CPG and CA both start from 97 of the 972 reference
points. Final point counts are 364, 486 and 729, including the coarse points.
CPG evaluates the complete frozen operator on the coarse input. CA uses the
128-to-4-to-1 stress head trained on coarse inputs of at most 97 points and
reuses pointwise first-block quantities during final prediction.

The experiment measures the same three profiling geometries (1800, 1899 and
1999), seeds 0, 1 and 2, and budgets as the earlier timing table. The 108
sampled-budget configurations and three full-mesh references each use 20
timed repetitions on one NVIDIA H20 in FP32 with batch size one. Method order
rotates between repetitions. Preparation is outside request latency, while
coarse inference, online density construction, sequential patch FPS,
Voronoi weights and output transfer are included. Full-field reconstruction
is measured separately and excluded from the manuscript latency table.

| Final budget | Uniform (ms) | PI (ms) | CPG (ms) | CA (ms) |
| --- | ---: | ---: | ---: | ---: |
| 37.5% | 0.994 ± 0.010 | 1.011 ± 0.011 | 3.115 ± 0.015 | 2.392 ± 0.013 |
| 50% | 1.043 ± 0.004 | 1.054 ± 0.005 | 3.363 ± 0.019 | 2.692 ± 0.022 |
| 75% | 1.167 ± 0.004 | 1.186 ± 0.002 | 4.012 ± 0.046 | 3.306 ± 0.022 |

The full-mesh reference is 1.256 ± 0.001 ms. Reported standard deviations
are across nine case/seed means, or three geometry means for the full mesh.
Within-case repetition variability is recorded separately in the raw data.

`verification.json` records identical point sets and full-field predictions
between the timed implementation and its CPU reference for all 111
configurations. The maximum absolute difference in relative-L2 error from the
saved accuracy evaluation is 1.104e-7. The original CPU nearest-neighbor
assignment is used for Voronoi weights, preserving its behavior at exact
distance ties; its computation and transfers are included in each request.

- `raw.jsonl`: per-configuration repetitions, stage measurements and reference comparisons.
- `configurations.json`: operator, head and complete frozen policy parameters.
- `environment.json` and `gpu_telemetry.json`: execution settings and GPU ownership measurements.
- `verification.json`: point-count, output-equivalence and saved-accuracy checks.

Reproduce from the paper repository with:

```bash
python scripts/benchmark_elasticity_cpg10.py \
  --adasolver-root ../AdaSolver_official --gpu 4 \
  --output ../elasticity_cpg10_revision/timing --sample-indices 0 99 199
python scripts/install_elasticity_cpg10_timing.py \
  --adasolver-root ../AdaSolver_official --run ../elasticity_cpg10_revision/timing
python scripts/build_timing_tables.py
```

The first command requires a new output directory and the experiment Python
environment used by AdaSolver. The implementation is in
`scripts/elasticity_timing/`. The installer validates the complete experiment
before replacing the Elasticity slices of the five timing CSVs.
