# Inference timing measurements

Source: `AdaSolver_official/inference_speed/profiles/all_methods_v4/final`.
This is the complete four-method profile, extending the earlier GPU v3 comparison.
The archived summaries were checked against the upstream reports on 2026-09-23.
Airfoil rows have been remeasured with the pressure policies in
`../airfoil_pressure/configs.json`, using the same cases, seeds, budgets,
and repetition counts. Their 111 raw case records are in
`../airfoil_pressure/timing_raw.jsonl`.
Elasticity rows are remeasured with the frozen 97-point CPG/CA policies in
`../elasticity_cpg10_timing/configurations.json`. The operator, 128-to-4-to-1
stress head, density parameters, point sets and reconstruction match the
accuracy evaluation. The 111 raw records and their numerical comparisons are
in `../elasticity_cpg10_timing/`.

The profiles cover the first, middle and last test case of eight benchmarks,
three sampling seeds, and point budgets 37.5%, 50%, 75%, plus the 100% mesh.
There are 888 measured case configurations. Each case is measured on a single
NVIDIA H20 in FP32 with batch size one; method order rotates between repeats.

- `summary.csv`: sampled-field and full-field latencies. `case_std_ms` is the
  sample standard deviation of nine case--seed mean latencies (three case
  means for the full mesh). `repeat_std_ms` describes repeat jitter separately.
- `accuracy.csv`: errors measured on those same profiling cases under each
  reconstruction rule. These are not full-test-set benchmark statistics.
- `stages.csv`: separately synchronized stage profiles. Their sum is not a
  substitute for the independently measured request latency.
- `preparation.csv`: setup components, outside repeated-query latency.
  Shared adapter preparation is recorded where it occurs and is not a method
  penalty. First-call timing already includes graph/JIT work; do not add its
  graph-setup value a second time.
- `shared_loads.csv`: shared data/model/head loading, separate from queries.

The GPU profiles use their recorded sampling implementations. Elasticity uses
97 of 972 coarse points (the 10% budget, rounded down) for both CPG and CA;
its final sets contain 364, 486 or 729 points, including the coarse set.
The CA stress head is trained on coarse inputs of at most 97 points.
Density construction, bounded integer allocation and sequential patch FPS run
on the GPU. Voronoi weights use the original CPU nearest-neighbor assignment,
including its behavior at equal distances; this computation and its transfers
are charged to every adaptive request.
Original reconstruction and the local
GPU reconstruction have separately measured errors; they are different numerical
operators on five tasks. Identical full-field entries for Elasticity,
ShapeNet-Car and DrivAerML are execution aliases with unchanged reconstruction.

Run `python3 scripts/build_timing_tables.py` from the paper directory to
regenerate the appendix inference latency table. Times remain in milliseconds;
DrivAerML is displayed to one decimal place, and other tasks to three.
The full-field reconstruction measurements remain in the archived CSVs but
are not included as a separate manuscript table.

To remeasure Elasticity, run `scripts/benchmark_elasticity_cpg10.py` with
`--adasolver-root` pointing to the code and checkpoint repository, `--gpu`
selecting an idle H20, and `--output` selecting a new run directory. Then run
`scripts/install_elasticity_cpg10_timing.py --run <directory>` with the same
`--adasolver-root`. The installer checks the complete case/seed/budget grid,
coarse and final counts, reference point sets, numerical outputs and saved
accuracy records before replacing the five Elasticity CSV slices. Other task
rows are preserved.
