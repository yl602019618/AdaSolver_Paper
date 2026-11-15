# Current-result reproduction

The standard cohort's original drivers/configurations are in the sibling E5_standard directory. These scripts independently audit or render the exported scalar results.

From the repository root:

```sh
python3 reproducibility/rebuttal/2026-11/current_results_20261115/darcy/validate_public.py --data-dir data/rebuttal/2026-11/current_results_20261115/darcy
python3 reproducibility/rebuttal/2026-11/current_results_20261115/reconstruction_airfoil/verify_export.py --data-dir data/rebuttal/2026-11/current_results_20261115/reconstruction_airfoil --output-dir /tmp/current_results_verification
```

Rendering reads CSV/CSV.gz only. Raw native-array audits require the external frozen workspace. Existing source-code commits, observed source hashes and result-package commits have different roles and are labelled accordingly in the audits.
