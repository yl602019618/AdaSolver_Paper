# LinearNO follow-up execution plan

Publication amendment, 2026-11-14: the user subsequently authorized pushing the reviewed current snapshot to an independent branch, with experiment types, changes and pending conditions documented. Root owns that publication; experiment workers do not issue Git writes. This supersedes the original no-push restriction below. Main, Feishu, the historical release and unrelated working-tree edits remain outside this publication.

The two authorized tasks are reconstruction optimization and matched Darcy/Airfoil extensions. All new files stay under `07_current/linearno_followup_20261114`. The historical release `91c1fa7cc81b8c1cb61a43618296c1c892d818c0` and current manuscript work are preserved. No remote push, main update or Feishu write is part of this task.

## Locked data and training

`standard_benchmarks/split_manifest.json` fixes a permutation of the original 1,000-case pool using seed20261114, with900 training and100 validation IDs. Normalization fits only those900 training IDs. Historical test populations have been viewed previously; the new cohort is validation-selected, not a claim of an untouched test set.

| Dataset | Main T3 / official point-temp comparison | Native grid reference | Evaluation population |
|---|---|---|---|
| Darcy | 85×85; 1000epochs; batch4; 225000updates; full7225 points; seed0 | Official conv-temp, Full only, same900/100 data | Existing200 smooth2 cases, explicit raw/global ID mapping |
| Airfoil | PressureQ[:,3],221×51;1000epochs;batch8;113000updates; variable4508–11271 points;seed0 | Official conv-temp, Full only; full-grid training separately labeled | Published110 cases with trueCl>0.02; optional all200 pressure population separate |

T3 uses the actual128-wide,8-layer,8-head,MLP-ratio2 configuration. The official standard-benchmark scripts use `conv_temp`,128 width,8 layers,8 heads,key dimension64 and MLP-ratio1. `conv_temp` reshapes to a fixed grid and is not used on subsets. Main sampling experiments use the existing official `LinearNO_temp` point variant, with the same named capacity settings and a distinct model identity. In this standard implementation `key_ratio=64` is an absolute key dimension, unlike the ShapeNet-Car implementation's multiplier.

Both tasks use AdamW and OneCycle with peakLR1e-3. T3 retains weight_decay1e-5 and max_grad_norm0.1. LinearNO uses the task's official defaults: Darcy weight_decay1e-6, Airfoil1e-5, both max_grad_norm1. No extra learning-rate search is planned. Darcy uses the shared physicalL2+0.1 derivative objective from the official Darcy runner; Airfoil uses shared decoded physical pressure relativeL2. These common objectives differ from the historical T3 normalized-target loss and are recorded as new matched protocols.

The native convolutional reference is a separate Full-only condition. Airfoil convolutional full-grid training is not silently treated as the same point-count schedule as the main variable-point cohort.

## Reconstruction and inference gate

The primary old/new comparison preserves each original rule:

- ShapeNet-Car: Euclidean k4 IDW, `1/(distance+1e-12)`, original exact-hit/tie behavior. Velocity uses the selected full cloud; pressure uses selected surface points and is reconstructed only to the surface.
- Darcy: original normalized coordinates first quantized tofloat32 and promoted tofloat64; QHull linear interpolation with nearest fallback.
- Airfoil: original integer index-grid coordinates; QHull linear interpolation with nearest fallback.

The shared entry point is `reconstruction/code/reconstruct_api.py`; accepted rules and code identity are in `reconstruction/accepted_protocols.json`. Final new-model evaluation additionally requires end-to-end validation with that checkpoint. Each saved prediction and point set is reconstructed with both old and new implementations for the full test comparison. New interpolation algorithms, if retained, have separate IDs and full reevaluation.

Primary scopeA is CPU input→complete CPU field, with FP32 network and float64 output container. ScopeB is GPU-resident input→complete GPU field and includes any required CPU work/transfers. ScopeC reports first construction and reuse of valid geometry/point-set maps, with identical cache opportunities for every method. Caches include physical geometry identity as well as reconstruction coordinates, selected indices, rule and dtype. Dynamic prediction/score caches are not reused across new inputs.

The main network implementation is held fixed while comparing reconstruction. The earlier LinearNO key-softmax layout optimization is not silently bundled into reconstruction speedup.

## Matrices and statistics

Each new main model has Full plus Uniform, PI, CPGβ0, CPGβ1 and its own trained CA head at37.5/50/75%, with sampling seeds0/1/2. CPGβ0/1 share coarse and final indices. CA cannot borrow the other backbone's head. Main rAUEC tables use CPGβ1;β0 is a separate mechanism table. Native conv Full has no rAUEC.

Darcy's primary PI uses the explicit2-D interface-distance correction, with prior amplitude0.25,ell2/84 andβ0.5 fixed before test. The historical flattened1-D EDT remains a distinct legacy configuration.

For each sampling seed, first average physical test errors over the fixed population at each budget, then integrate the three-point curve and divide by that same checkpoint's Uniform area. Finally average the three ratios and use sampleSD(ddof1). Full100% is outside this integral. Missing CA or failed groups remain N/A with reasons.

Formal timing uses at least3 fixed cases per task/architecture,20 warmups and100 repetitions per condition, directly measured wall time with boundary synchronization. Stages are profiled separately. Old/new and both backbones use the same uncontended GPU and recorded interleaving; memory is separately measured with declared allocator/external-library scope.

## Owners and resources

| Owner | Deliverables | Initial GPU plan |
|---|---|---|
| followup_reconstruction | Car full100-case old/new accuracy, A/B/C timing, shared API, correctness gate, reconstruction-only profiles and figures | GPU6 timing;7 validation when available |
| followup_darcy | New T3/point-temp/conv training, own CA heads, complete main matrix and A/B/C costs | GPU0/1/5, wait for actual availability |
| followup_airfoil | New pressure T3/point-temp/conv training, own CA heads, complete main matrix and A/B/C costs | GPU2/3/4 |
| root | Protocol/source registry, shared reconstruction handoff, independent statistics, combined paper-style tables/figures and archival package | CPU coordination |

An unrelated session launched fixed-set evaluation and then standard-model training after the initial idle-GPU snapshot. Its work is not terminated. Airfoil's early overlap is recorded; final latency remains exclusive. Darcy and reconstruction wait for their designated cards while CPU implementation continues. Changes to resource policy are recorded separately from scientific configurations.

## Completion gates

1. Training reaches the locked updates and validation-selected checkpoint; model,normalizer and head identities are recorded.
2. Reconstruction rule equivalence and per-new-checkpoint end-to-end validation pass; otherwise classify a new algorithm protocol.
3. Every expected case/seed/budget/method appears, including failure records; CPG point identities and Full/Uniform references are checked.
4. Independent rAUEC recomputation agrees; A–D tables and PDF/SVG/300dpi PNG are generated from the checked CSVs.
5. The local `archive_ready/E1_reconstruction` and `archive_ready/E3_standard` packages record complete/partial scopes and commands. No missing cell is filled from an old unmatched model.
