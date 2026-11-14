# Airfoil seed0 accuracy

This group contains one trained Transolver3 model and one official LinearNO point-temp model. Each model has its own pressure head. Training uses900 cases, validation uses100 cases, and evaluation uses the published110 positive-lift cases on the221×51 grid. The test population has historical exposure; it is not a newly sealed holdout.

Both models completed1000 epochs and113000 updates. Full, Uniform, PI, CPG beta0/beta1 and CA cover all110 test cases; adaptive methods use budgets37.5%,50%,75% and sampling seeds0/1/2. Every old/new reconstruction row uses the same predictions and selected indices. Each model contributes5060 distinct prediction requests and40480 long metric rows.

The CSVs contain both reconstruction protocols, distinguished by `reconstruction_id`. Force errors are diagnostics from the same field-target policies. Primary CPG uses beta1; beta0 is a paired compensation ablation. Sampling SD is the sample SD across three sampling-seed case means. Full has no sampling seed. No training-seed SD is reported because there is only one independent model per backbone in this group.

`audit.json` records the model/head/normalizer/source hashes, complete matrix checks, raw-to-metric replay, and rAUEC recomputation. Native long metrics matched the raw observations exactly. The largest rAUEC reaggregation difference was1.42e-14. The existing full-array acceptance was reused, with138 independent NPZ spot checks per model.

Training seeds1/2 and formal complete-output latency remain separate pending work. Existing other-session controllers own the latency queue. These accuracy values are not paired with old-checkpoint times or fixed-map microbenchmarks.

Rebuild figures from the included CSVs:

```sh
python reproducibility/rebuttal/2026-11/linearno_followup_20261114/plot_airfoil_seed0.py --data-dir data/rebuttal/2026-11/linearno_followup_20261114 --output-dir figures/rebuttal/2026-11/linearno_followup_20261114/airfoil_seed0
```

Native raw arrays and checkpoints remain external; their relative paths and hashes are recorded in airfoil_seed0_audit.json. Public CSV statistics are reproduced by the package validator.
