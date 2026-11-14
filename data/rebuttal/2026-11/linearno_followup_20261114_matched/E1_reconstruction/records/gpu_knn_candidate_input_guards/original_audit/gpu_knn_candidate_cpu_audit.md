# GPU kNN candidate: independent CPU oracle audit

The candidate passes all 12 selected valid-domain CPU cases, including exact source-neighbor coverage, native tie ordering, epsilon behavior, sparse source sets, and multichannel application. The largest observed field difference is **2.220446049250313e-16**. Four classes of input-contract defects were reproduced separately. This evidence does **not** establish CUDA correctness, measured GPU memory use, speed, or end-to-end acceptance. The candidate remains separate from the accepted reconstruction API v1.1.

## Identity and reproducibility

- Audited source: [`../code/gpu_knn_candidate.py`](../code/gpu_knn_candidate.py).
- Source SHA256: `c79f2352fc3a9a29518fe52b1e38985c969f13c0b1c30cbff1695cb5891f8035`.
- Audit completion: `2026-11-14T13:09:07.808363+00:00`.
- Independent audit script: [`gpu_knn_candidate_cpu_audit.py`](gpu_knn_candidate_cpu_audit.py).
- Machine-readable measurements and issue probes: [`gpu_knn_candidate_cpu_audit.json`](gpu_knn_candidate_cpu_audit.json).
- Environment: Python 3.10.12, PyTorch 2.10.0+cu128, NumPy 2.2.6, SciPy 1.15.3; all tensors on CPU and `CUDA_VISIBLE_DEVICES=''`.
- The candidate source hash was checked before and after the run and remained unchanged. Neither the candidate nor the frozen API was edited by this audit.

Run from `07_current/`:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 120-123 python3 \
linearno_followup_20261114/reconstruction/results/gpu_knn_candidate_cpu_audit.py
```

The oracle independently evaluates SciPy `cKDTree` on the original source order, then applies the native inverse-distance rule. Its output is checked for exact equality against `reconstruct_api.native_idw` before comparing the candidate. The audit uses fixed RNG seeds `20261114` for valid cases and `782` for failure probes.

## Coverage and numerical results

The 12 cases contain 244 query rows, 515 output elements, and 109 blocked `cdist` calls. All tested neighbor-index arrays match the native oracle exactly. Maximum relative field norm difference is **1.0947474572250936e-16**. There are **0** violations of `abs(new-old) <= 1e-6 + 1e-5*abs(old)`; the audit additionally requires `rtol=5e-13, atol=2e-12` for field values and `rtol=5e-14, atol=1e-14` for neighbor distances. Those stricter checks were fixed in the script before execution and all passed. Outputs in this candidate are float64, including when source values are FP32.

| Case | Sources / queries | Effective k | Channels | Native fallback rows | Maximum field difference |
|---|---:|---:|---:|---:|---:|
| Random 2D scalar, query blocking | 37 / 23 | 4 | 1 | 0 | 0 |
| Random 3D, FP32 values, query blocking | 41 / 17 | 4 | 4 | 0 | 2.220446049250313e-16 |
| Duplicate coordinates and exact hits | 6 / 4 | 4 | 3 | 4 | 0 |
| Permuted duplicate source order | 6 / 4 | 4 | 3 | 4 | 0 |
| Equal-distance group at cutoff | 8 / 2 | 3 | 1 | 2 | 0 |
| k=1 tie with vector values | 8 / 2 | 1 | 4 | 1 | 0 |
| Fewer sources than requested k | 2 / 3 | 2 | 2 | 1 | 0 |
| Single source, scalar values | 1 / 2 | 1 | 1 | 0 | 0 |
| Below / at / above epsilon | 4 / 3 | 4 | 1 | 3 | 0 |
| Offset 1e12 with close coordinates | 5 / 2 | 4 | 1 | 0 | 0 |
| Integer-grid ties | 67 / 169 | 4 | 2 | 156 | 0 |
| Noncontiguous coordinate arrays | 23 / 13 | 4 | 3 | 0 | 3.469446951953614e-17 |

For each observed block, the audit verifies that `cdist` receives the complete original source tensor and that concatenating the query blocks recovers every query exactly once, in order. Thus the implementation examines every source for every query in these cases; it does not depend on a heuristic neighbor-candidate radius. This is exhaustive **source coverage**, not an exhaustive test of all representable float64 inputs. The configured large task sizes below were analyzed from the block formula, not executed as large CPU tests.

The implementation requests up to `effective_k + 1` neighbors. This additional neighbor exposes equal distances that cross the k-th cutoff. Any near-equal adjacent distances among the retrieved neighbors trigger native `cKDTree` lookup with the original source order and the original `effective_k`. The cutoff, k=1, duplicate, and permuted-source cases confirm this behavior on CPU. The integer-grid case sends 156 of 169 rows (92.31%) through the native fallback; 171 of 244 rows across the deliberately tie-heavy audit use fallback. These rates demonstrate why runtime must be measured on actual task geometry and cannot be inferred from the GPU search implementation alone.

The distance is ordinary Euclidean distance, not squared distance. Application uses `1 / (distance + 1e-12)` and chooses the first native returned neighbor at distance `<= 1e-12` for an exact hit. The epsilon case includes values immediately below, equal to, and above this boundary. Scalar inputs return `[N_query, 1]`; vector inputs return `[N_query, C]`. This matches the frozen plan-level `native_idw` interface. The separate historical `knn_idw` helper can squeeze scalar outputs, which is not treated as a defect in this interface.

## Confirmed input-contract defects

| Finding | Reproduction | Consequence | Required disposition |
|---|---|---|---|
| `max_pairs` is not enforced when it is smaller than the source count | 11 sources with `max_pairs=10` produce an 11-pair tile; `max_pairs=0` is also accepted | A query-block floor of 1 violates the advertised pair cap | Validate a positive budget and reject `N_source > max_pairs`, or implement source-axis blocking with equivalent ties |
| Optional CPU coordinate copies are not checked against the tensors | Reverse `source_cpu` with unchanged tensor inputs: native output 9, candidate output 30; stale `query_cpu`: native output 9, candidate output 40 | Tie fallback can query a different geometry/order and apply its indices to the original values without an error | Derive fallback copies from tensors, or enforce their shape, value, and ordering identity before use |
| Invalid k is accepted by the builder | `k=0` and `k=-1` return maps; `apply` later raises `IndexError: argmax(): Expected reduction dim 1 to have non-zero size.` | Failure occurs after returning an invalid mapping | Require a positive integer k before search |
| Nonfinite source coordinates differ from native rejection behavior | One NaN source row causes native `cKDTree` to raise `ValueError`; candidate returns a map with finite retained distances | A bad source can be silently excluded by top-k | Validate finite coordinates or enforce a checked finite-input contract at the caller |

The CPU-copy failures are conditional on supplying inconsistent optional inputs. The normal cases cover both supplied matching copies and copies derived by the candidate. The pair-cap failure is a real API boundary defect, but it does not arise at the audited configured task sizes with the default `max_pairs=8,388,608`. None of these probes modifies accepted v1.1 reconstruction results.

## Pair-tile bounds for current sizes

The implementation uses `query_block = max(1, min(512, max_pairs // N_source))`. With the default pair budget, the following arithmetic bounds hold for one float64 distance tile:

| Source points | Query points | Query block | Pairs in maximum tile | Tile MiB |
|---:|---:|---:|---:|---:|
| 2,709 | 7,225 | 512 | 1,387,008 | 10.58203125 |
| 8,453 | 11,271 | 512 | 4,327,936 | 33.01953125 |
| 24,140 | 32,186 | 347 | 8,376,580 | 63.908233642578125 |
| 32,186 | 32,186 | 260 | 8,368,360 | 63.84552001953125 |

These values describe one distance tile only. They are not peak allocator measurements: live tensors, top-k workspace, retained map arrays, transfer buffers, and transient allocations add memory. The loop still performs `N_query * N_source` distance evaluations. CPU fallback and transfers can remain substantial even though the full distance matrix is not retained.

## Acceptance boundary

This audit supports the numerical behavior of the separate candidate on the selected finite CPU cases. Before promoting it to an accepted GPU protocol, resolve or enforce the stated input contracts, repeat the relevant checks on CUDA, and validate the complete reconstruction/request path with fixed checkpoint predictions and retained indices. The current audit does not test kept-point scatter outside this mapper, complete velocity/pressure/Cd reconstruction, downstream field metrics, CPU-to-CPU request timing, caching behavior, or real peak GPU memory. It therefore supplies no speedup claim and does not replace the existing v1.1 acceptance evidence.
