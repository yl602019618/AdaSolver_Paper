# E4 derived analysis reproduction

`recompute_e4_analysis.py` verifies the original common/native CSV matrix and recomputes the new derived tables using public CSVs. It requires Python3; Matplotlib is used only if `--plot` is requested. It does not load model weights or run inference.

```bash
python3 reproducibility/rebuttal/2026-11/E4/recompute_e4_analysis.py --data-dir data/rebuttal/2026-11/E4 --output-dir /tmp/e4-derived-check
```

The public `derived_analysis/region_energy_inputs.csv` records region relative errors and the corresponding truth-region/full norms. These scalars reconstruct region contributions and their cross-resolution changes. Their original extraction used the declared421² cached source data; rederiving masks and norms from the original fields requires those data. Public CSV verification does not claim to recheck private point arrays. The18 original coarse-prefix checks are retained in the independent audit.

`source_workflow/` preserves the actual development-machine analysis and original frozen inference driver. Those scripts depend on the original workspace layout, datasets and model indices. They explain provenance and full reproduction; they are not the portable public verification entry point.

The original27,000 requests are reused. The update has one training seed and three sampling seeds. No new accuracy run or formal repeated timing is performed.
