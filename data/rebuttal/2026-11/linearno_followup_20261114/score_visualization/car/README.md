# ShapeNet-Car acquisition visualization

The fixed case is `param0/100715345ee54d7ae38b52b4ee9d36a3`, the first locked test case. All views use E5 Transolver3 training seed0, its own head seed0, sampling seed0 and50% budget. Selection contains16,093 of32,186 native points.

`figures/car_score_density_selected` shows the exact archived CPG/CA refinement scores, genuine Uniform/PI priors, paper-defined designed target densities, and actual saved points. Its inputs are the four `data/*_perpoint.csv` files. Per-point order columns preserve the original selected/coarse ordering. CPG and CA each retain3,218 of3,218 actual coarse points. Uniform and PI have no model-acquisition coarse set.

PI uses a hard shell at distance0.1, with configured half-uniform/half-shell allocation. It has no separate native normalized-score field. Its displayed target is explicitly derived from the frozen mixture configuration and the locked paper formula; it is not an array emitted by the native selector. Integer rounding gives8,047 uniform-base and8,046 extra points. The actual run needs4,041 nonshell fallback points. Native selected compensation density is preserved separately from the designed full-domain target.

For CPG and CA, `target_density` normalizes the actual native priority `(1+64 exp(-d/0.1)) (1+2 z_score)`. The companion `target_reference_ratio` has mean1 under the uniform reference cloud. `allocation_density` is the actual integer stratum quota divided by reference count; `normalized_allocation_density` is that quota ratio divided by16,093. Neither measure is an independent point-inclusion probability under mandatory anchors and FPS. The raw selected compensation density and applied weights are separate columns. The fixed paper formulas and SHA are recorded in `score_manifest.json`.

CPU intermediate replay passes the unchanged elementwise archived-score tolerance for CPG. CA has154 violations at `atol=1e-6, rtol=1e-5`, with maximum absolute difference `4.05517e-4`. The saved GPU CA score was never replaced or altered. The original GPU raw-head replay remains `PARTIAL`.

`ca_cpu_diagnostic` is a separate complete native CPU run. Its own256-dimensional block0 features, raw head output, pressure proxy, raw/normalized score, native quota/FPS selection, final prediction and error form one chain. Ground truth is first used after final prediction, solely for error. Its ordered selected indices happen to match the GPU run exactly, while its score parity difference remains154 points. Its field error is0.0676564730 and its1.783s wall time is a one-case diagnostic cost, not a latency benchmark. `ca_cpu_diagnostic/figures/car_ca_cpu_complete_chain` displays the chain from feature summary to its own error.

The full vectors and prediction caches in NPZ remain internal, with hashes. User-authorized per-point CSVs contain coordinates, score/density, order, feature summaries, predictions and error diagnostics; they contain no complete CFD truth field. Every figure reads only CSV and uses the same fixed camera and geometry-only display rule: native surface plus a central slab of volume points. No coordinates are jittered or re-meshed.

CSV-only figure reproduction:

```sh
python plot.py --data-dir data --output-dir figures
python plot_cpu_ca.py --csv ca_cpu_diagnostic/perpoint.csv --output-dir ca_cpu_diagnostic/figures
```

Native-source reproduction from a preserved workspace, into new directories. Copy the scripts into the new Car directory first; the derivation receipt binds its colocated capture script.

```sh
mkdir -p /path/to/new/car
cp capture.py /path/to/new/car/capture.py
CUDA_VISIBLE_DEVICES='' python capture.py --workspace /path/to/07_current --output /path/to/new/car
CUDA_VISIBLE_DEVICES='' python cpu_ca_run.py --workspace /path/to/07_current --output /path/to/new/car/ca_cpu_diagnostic
python derive_allocation.py --workspace /path/to/07_current --output /path/to/new/car
python prepare_public_sources.py --output /path/to/new/car
```

`gpu_raw_replay.py` is prepared but has not run. It exits immediately if the selected device is occupied and requires a fresh output directory. There is no additional GPU worker or wait process from this visualization task. An optional later replay command, after resource coordination, is:

```sh
python gpu_raw_replay.py --workspace /path/to/07_current --gpu DEVICE_INDEX --output /path/to/new/car_gpu_raw_replay
```

The initial reuse decision is preserved in `../car_reuse_audit.json`; the native acquisition facts are in `score_manifest.json`; the distinct CPU run is in `ca_cpu_diagnostic/run_manifest.json`. Independent QA is maintained by the separate score-audit worker.

The initial capture did not record its script hash before execution; that original provenance gap remains explicit. A new isolated capture under `replay_validation` recorded170 source/model/normalizer hashes before execution and verified them afterward. All89 base arrays exactly reproduced the initial capture. Its before-execution receipt and array replay validation support current reproducibility without inventing an old execution hash.
