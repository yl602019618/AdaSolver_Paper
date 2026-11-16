# PI beta calibration: independent protocol audit

Audit date: 2026-11-17. This was a read-only audit of the existing implementation, model identity, published E4 records and historical beta results. No GPU, model inference, training, remote action or modification of existing results was performed.

## Verified baseline identity

- E4 uses a single Transolver3 checkpoint trained on the full 85×85 grid for 1,000 epochs, training seed 0.
- Actual checkpoint SHA-256: `5c8a7fa3d24a9ead213144e8a9a1ca3a932bd5c8265e504e92c4d7120175c1fb`.
- Actual output-normalizer SHA-256: `72b841a23d46e83d41430d365f0b57c73b5ed87864acda3cb7efd800ed64b16e`.
- The E4 training-derived global normalizers are coefficient mean `7.498525142669678`, std `4.5`; solution mean `0.005595447029918432`, std `0.0038398094475269318`.
- Observations are true stride 5/3/2 nodal selections from the 421² source fields, giving 85²/141²/211². The primary metric is evaluated on the common true 211² solution grid.
- Existing E4 PI uses `interface_distance`, amplitude `0.25`, length `2/84`, beta `0.5`, budgets `(0.375,0.5,0.75)` and sampling seeds `(0,1,2)`.
- Preserve the original flattened-mask 1-D EDT for a beta-only experiment. Replacing it with the later physical 2-D EDT would be a separate strategy change.

Source paths relative to `07_current/`:

- `rebuttal_20261112/experiments/E4_darcy_resolution/protocol.md`
- `rebuttal_20261112/experiments/E4_darcy_resolution/config/protocol_lock.json`
- `rebuttal_20261112/experiments/E4_darcy_resolution/code/run_resolution.py`
- `AdaSolver_official/checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_manifest.json`
- `AdaSolver_official/adasolver/inference/pi/darcy.py`
- `AdaSolver_official/adasolver/utils/seed.py`

## Beta changes compensation only

The actual implementation computes

`w_i = 1 + 0.25 * exp(-distance_i / (2/84))`,

then samples `rng.choice(N, K, replace=False, p=w/sum(w))`. Only after selecting the indices does it compute

`compensation = (1 / w[selected]) ** beta`, followed by a float32 cast.

Beta therefore does not change the proposal or retained indices if the same input and fresh RNG state are used. The RNG is NumPy `SeedSequence([sampling_seed, global_case_id])`.

The variable named `relative_weight` in the source is the **raw prior weight**, not normalized probability. Do not recover new weights from the already rounded float32 beta=0.5 compensation, or silently substitute normalized density. Keep the original float64 prior weights and apply each exponent before the float32 cast, or call the native sampler with identical fresh RNG state for each beta. Record and compare actual index hashes.

With this prior, weights lie in `[1,1.25]`, and beta=2 compensation lies in `[0.64,1]`. The code allows nonnegative beta without an upper bound. Beta>1 is an over-correction candidate rather than tempering in `[0,1]`.

Sampling is weighted without replacement. The inverse proposal factor is a nominal correction, not the exact final inclusion probability. Increasing beta is not guaranteed to monotonically improve the final field error, particularly at large retained fractions.

## Calibration population

The proposed calibration set, smooth2 rows **400:500**, maps to global case IDs **1400:1500** (end exclusive). It is disjoint from all the following declared populations relevant to this comparison:

- Target E4 checkpoint optimization: smooth1 rows 0:1000.
- Target E4 checkpoint selection and final comparison: smooth2 rows 0:200, global IDs 1000:1200. The checkpoint manifest explicitly says `test_used_for_selection=true`.
- Historical PI calibration: smooth2 rows 0:12; historical PI confirmation: rows 12:72; final old PI evaluation: rows 0:200.
- Separate standard three-training-seed Darcy validation: smooth2 rows 200:400, global IDs 1200:1400.
- New 900/100 matched Darcy comparison: train/validation both come from the original smooth1 1,000-case pool; test is smooth2 rows 0:200.

Rows 400:500 can therefore be described as a **calibration population disjoint from the E4 test cases**. This audit does not establish that nobody in the wider project has ever viewed these rows, and it does not erase the target checkpoint's historical test-informed selection. Continue to disclose that history.

The final E4 test population must remain rows 0:200. Calibration targets must not enter model normalization fitting or alter the frozen checkpoint. Use original training normalizers at all resolutions.

## Existing reusable evidence

`beta_ablation/results/darcy_pi/` contains a completed historical beta ablation at the **native 85² evaluation grid**. Each beta has 1,800 records, covering 200 cases, three budgets and three sampling seeds. Independent read-only checks found:

- all five beta files have exactly the same 1,800 condition identities;
- all selected-index hashes agree across beta=0/0.25/0.5/0.75/1;
- those 1,800 hashes also agree with E4's 85² PI records;
- the beta=0.5 native per-case errors differ from E4's native-grid diagnostic by at most `1.3018522985325243e-8`;
- the old interpolation-cache manifest reports an exact native output check with maximum difference 0.

Seed-first native-grid rAUEC recomputed from those existing summaries:

| Historical beta | Mean rAUEC | Sampling SD |
| --- | ---: | ---: |
| 0 | 1.0314595853 | 0.0165433985 |
| 0.25 | 1.0087958879 | 0.0170153876 |
| 0.5 | 0.9964534237 | 0.0177322886 |
| 0.75 | 0.9938857742 | 0.0186468521 |
| 1 | 1.0002853323 | 0.0196998934 |

These are already viewed **historical test sensitivity results**. They are neither new calibration data nor common-211² results. The old beta=0.75 native mean being below one does not imply each sampling seed is below one, or that all three E4 resolutions can be made below one. No reusable 141²/211² beta sweep was found; E4 has only its fixed beta=0.5 PI condition at those resolutions.

## Assessment of the proposed selection rule

The parent's predeclared grid `[0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.5,2]` is compatible with the actual sampler. A common beta selected to minimize the maximum **mean calibration rAUEC across the three resolutions** is a coherent criterion for one shared cross-resolution configuration. Define ties before seeing the new calibration outcomes, e.g. closest to 0.5 and then smaller beta.

For every calibration beta and sampling seed: first average the full-grid physical errors over cases at each budget, integrate the three-point curve, divide by the same resolution/checkpoint/sampling-seed Uniform area, and finally average the three sampling-seed ratios. Full does not enter rAUEC.

Per-resolution validation-optimal betas may be reported as a separately named resolution-conditioned diagnostic, if all three are frozen before final-test evaluation. Report this alongside the common-beta result rather than choosing between them retrospectively based on test outcomes. Preserve all validation candidates, the selected-beta final results, and the beta=0.5 baseline, including any value at or above one. Mean rAUEC below one and every sampling-seed rAUEC below one are distinct statements.

The old published numbers should remain as the frozen baseline. New calibrated results need a new protocol identity and a clearly stated selection population. A result below one is an empirical outcome to check, not a guaranteed acceptance condition or a reason to continue selecting against the final test set.
