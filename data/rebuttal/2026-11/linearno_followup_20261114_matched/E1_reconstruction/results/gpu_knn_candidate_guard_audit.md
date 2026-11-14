# GPU-kNN candidate input-guard follow-up

Candidate version: `0.2-input-guards`; SHA256: `47e77fa402622f14583f814ef04b4ffe3731f28f1f6a124474ae2427e0f594b0`.
The unchanged independent valid-domain suite was rerun:12 cases,244 queries,515 output elements; maximum field difference 2.2204460492503131e-16, with0 violations of the declared elementwise tolerance. All neighbor indices still match the native oracle.
All29 invalid-input probes are rejected by the builder before any torch.cdist call. Positive NumPy integer arguments, max_pairs exactly equal to the source count, matching Torch CPU side copies, and validated side-copy snapshot behavior also passed.

| Fixed defect | New behavior |
|---|---|
| Pair budget below source count or nonpositive | Reject before search; one query must fit the exhaustive source axis |
| Inconsistent optional CPU copy | Check float64 dtype, exact shape, values and point order against the tensor, then retain a snapshot; mismatches fail even on non-tie queries |
| Invalid k | Require a positive integer; reject zero, negatives, floats and booleans before search |
| Nonfinite source/query | Reject NaN and ±infinity before search, matching the finite native geometry contract |

The positive pair-budget guard bounds one distance tile only; retained indices/distances, top-k workspace and other live allocations still add memory. With supplied CPU copies, exact tensor-copy identity checks may transfer coordinates from GPU and synchronize; this work belongs to any future candidate timing and is not hidden as free preparation.

The original source and audit are preserved under code/versions/gpu_knn_candidate_before_input_guards.py and records/gpu_knn_candidate_input_guards/original_audit/. The original results/gpu_knn_candidate_cpu_audit.{py,json,md} are unchanged. The new runner is code/audit_gpu_knn_guards.py.

This rerun was executed by the implementation follow-up thread using the original independent valid-case suite. It is CPU correctness evidence only. No CUDA validation, peak GPU allocation measurement, end-to-end acceptance or speedup claim is made. The candidate remains separate from frozen API v1.1, and the existing Car controller is unchanged.

Reproduce from07_current:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c128-135 python3 linearno_followup_20261114/reconstruction/code/audit_gpu_knn_guards.py
```
