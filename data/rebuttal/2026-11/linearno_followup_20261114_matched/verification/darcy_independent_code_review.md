# Darcy independent evaluation review

Reviewed by the root coordinator on 2026-11-14: `evaluate.py`, `finalize_accuracy.py`, `inference.py`, and `ca_support.py` in `standard_benchmarks/darcy/code/`.

The implementation computes each seed's case-average curve before integrating, divides by the same checkpoint/seed Uniform area, then reports mean and sample SD. Full is excluded from the integral. The finalized matrix checks the locked test IDs and budgets; the convolution model remains Full-only. CPG beta=0 and beta=1 share coarse prediction and retained indices. Old/new reconstructions consume the same predicted values. Both Full reconstruction rows reference one physical Full request.

The review found that CA loading checked the operator and head weights but omitted an explicit normalizer identity comparison. This has been fixed; `standard_benchmarks/darcy/results/head_normalizer_identity_validation.json` records the matching case and rejection of mismatched identities. Runtime also compares the backbone normalizer with the actual current normalizer file.

The stored Darcy coarse prediction is in normalized output units, matching the native grad_lap acquisition path. Final predictions are decoded before reconstruction and physical relative-L2 evaluation. The units are now stated in the protocol and raw-run metadata. The Transolver3 feature tap's unconditional placeholder addition matches the actual model's forward implementation.

This is a source-level review. It does not certify a finished trained matrix or measured GPU latency; those remain separate runtime receipts. No training process was restarted for these changes.
