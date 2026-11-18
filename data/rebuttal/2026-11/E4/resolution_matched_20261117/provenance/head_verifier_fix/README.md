# Head acceptance repair, 2026-11-19

All three backbones and all three own-resolution CA heads had completed their full 1000 epochs. The pipeline stopped in the post-training verifier: the stored FP32 feature bank was generated in batches of eight, while its replay used one case per forward. At 85/141/211 resolutions, the original case-0 feature check reported maxima 2.29e-5 / 2.29e-5 / 5.34e-5 and 346 / 1179 / 14289 elementwise violations.

The verifier now replays the actual bank batches [0,8) and [896,904), using the same checkpoint, normalizer, coordinates, retained indices, batch order and full-forward block-0 hook. Every bank feature and prediction check is exactly equal at all three resolutions. It also checks the actual batch-one inference feature extractor and head against the batch-one full-forward hook; these checks are exactly equal too. The original elementwise tolerance, abs difference <= 1e-6 + 1e-5 * abs(reference), is unchanged.

Batch-one versus batch-eight differences remain visible in each head receipt under `batch_size_sensitivity_diagnostics`, including quantiles and relative norms. Those comparisons describe different FP32 execution shapes; they are separate from checks of a bank replay under its recorded context. No model weights, training data, normalizer, training rule, sampling rule or reconstruction rule changed. No head/backbone retraining was needed.

`verify_head_before.py` preserves the previous verifier. `fix_receipt.json` links the original errors, old/new code hashes and all three successful production verification runs. The final accepted receipts are `results/head_r85_validation.json`, `results/head_r141_validation.json`, and `results/head_r211_validation.json`. Evaluation and publication remain parent-pipeline responsibilities.
