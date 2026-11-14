# Score visualization reproduction

The plotting scripts read only the published scalar CSVs. No model or truth file
is needed to redraw the figures. From the repository root:

```sh
python3 reproducibility/rebuttal/2026-11/linearno_followup_20261114/score_visualization/airfoil/plot_scores.py --data-dir data/rebuttal/2026-11/linearno_followup_20261114/score_visualization/airfoil --output-dir /tmp/airfoil_score_figures
python3 reproducibility/rebuttal/2026-11/linearno_followup_20261114/score_visualization/car/plot.py --data-dir data/rebuttal/2026-11/linearno_followup_20261114/score_visualization/car/data --output-dir /tmp/car_score_figures
python3 reproducibility/rebuttal/2026-11/linearno_followup_20261114/score_visualization/car/plot_cpu_ca.py --csv data/rebuttal/2026-11/linearno_followup_20261114/score_visualization/car/ca_cpu_diagnostic/perpoint.csv --output-dir /tmp/car_cpu_ca_figures
python3 reproducibility/rebuttal/2026-11/linearno_followup_20261114/score_visualization/validate_score_public.py --data-dir data/rebuttal/2026-11/linearno_followup_20261114/score_visualization
```

Dependencies and rendering versions are in the parent reproducibility folder.
Full native-array QA uses `validate_score_portable.py`; see `PORTABLE_QA.md`.
It requires the external frozen workspace and dataset paths explicitly supplied.
Capture scripts are source snapshots: Car accepts `--workspace` and `--output`;
the Airfoil capture retains its original workspace layout, so run it from
`linearno_followup_20261114/reuse_review/score_visualization/airfoil` in a copied
external workspace. Capturing raw inference is separate from CSV-only plotting.
Weights, original CFD truth and full feature tensors are external, with hashes
and relative acquisition locators in the source manifests.
