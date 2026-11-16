# Verification and reproduction

From the repository root, recompute the public records and subset statistics:

```sh
python3 reproducibility/rebuttal/2026-11/p0_airfoil_original_cdcl_seeds123_posthoc_20261117/verify_posthoc.py --data-dir data/rebuttal/2026-11/p0_airfoil_original_cdcl_seeds123_posthoc_20261117
```

The actual fixed-seed protocol, launch and pipeline receipts are included as extension_*.json. Training source/configs are under the accompanying reproduction directory. They use Python3.10.12, PyTorch2.10.0+cu128, CUDA12.8, the same P0 cache/normalizer and external checkpoints identified in the registry. The full native seed 3 commands are in extension_pipeline_status.json. The original Cd/Cl evaluator and reference files remain in reproducibility/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/. Dataset and checkpoint path bindings refer to the recorded workspace. No seed 0 data are deleted or relabeled.
