# Airfoil refinement-score diagnostic

The fixed case is Airfoil1001, Transolver3 training seed0 with its own learned CA head, sampling seed0 and budget50%. The four actual selections contain5636 of11271 points. All selected-index orders, coarse-index orders and compensation values match the saved native records exactly.

- [Raw and normalized scores](figures/airfoil_score_stages.png)
- [Proposal, empirical selection density and actual points](figures/airfoil_sampling_comparison.png)
- [PI policy score, CPG coarse pressure and CA decoded head pressure](figures/airfoil_acquisition_inputs.png)
- [Post-hoc error only](figures/airfoil_error_only.png)
- [Surface-arclength score and density curves](figures/airfoil_surface_curves.png)
- [CA latent feature summary and standardized head output](figures/airfoil_ca_feature_and_head.png)

Each figure also has PDF and SVG versions with a caption. Scores and sampling were captured on CPU by observing the actual frozen functions. No model was trained, no occupied GPU was used, and CFD truth was unavailable to the score graph. Uniform has no learned refinement score. PI uses a geometry policy score; its sum-normalized prior is separate from the actual eligible-point proposal.

`score_manifest.json` states each field's support, normalization and source function. `truth_isolation.json` records the negative controls and actual access log. Every mandatory surface point has zero extra-draw proposal probability because it is already selected; this does not mean zero final allocation. The surface curves plot these zeros without alteration.

The raw128D CA first-block features and8D bottleneck remain in the external archive. Public scalar CSVs include their L2/mean/std summaries and the raw standardized pressure-head output. These latent summaries are distinct from decoded physical pressure and its downstream gradient acquisition score. All raw CFD truth arrays remain external. `public_data_manifest.json` contains the public CSV and external-asset hash index.

To redraw all figures using only the public CSVs:

```sh
python3 plot_scores.py --data-dir . --output-dir figures
```

A CSV-only replay was checked in `offline_replay_validation.json`. The plot requires NumPy, Matplotlib and Pillow; no model, checkpoint or CFD dataset is needed.

Data-source QA passes within this single-case CPU diagnostic. Paper consistency remains PARTIAL: the original weighted CPG sampler retains619/1127 coarse points, whereas weighted_nested CA retains1127/1127. Filled orange marks are retained coarse points and orange crosses are omitted coarse points. This diagnostic preserves the original selectors and does not establish the paper's general nested-S0 statement. See `protocol_differences.md`.

Original native GPU raw-score values were not saved. The new values are a CPU observer capture of the same checkpoint and scientific functions; selected-set and compensation replay is exact, while bitwise native raw-score equality is not claimed. Instrumented CPU capture times are not online GPU latency measurements.
