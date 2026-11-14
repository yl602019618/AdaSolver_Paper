# Darcy follow-up: execution status

The data, matched training plan, official model identities, reconstruction contract and executable pipeline are prepared. No new matched Darcy checkpoint or final accuracy/latency result has been produced yet: GPU 0, 1 and 5 are occupied by external jobs. The waiting controllers preserve those jobs and require a stable idle interval before admission. Live progress is in `read.me` and `runtime/`.

| Backbone | Actual variant | Parameters | Planned training | Full field error | rAUEC | Formal latency |
| --- | --- | ---: | --- | --- | --- | --- |
| Transolver3 | Native `Transolver3Adapter` | 977,089 | 1,000 epochs, seed 0, 900/100 split | N/A: waiting for training | N/A: waiting for evaluation | N/A: waiting for accepted trained output |
| LinearNO | Official `LinearNO_temp` point blocks | 585,473 | Same split, epochs, update count and point schedule | N/A: waiting for training | N/A: waiting for evaluation | N/A: waiting for accepted trained output |
| LinearNO | Official `LinearNO_Conv_temp` | 1,766,145 | Same matched training; Full only | N/A: waiting for training | N/A: variable-point interface unsupported | N/A: waiting for accepted trained output |

The original smooth1 pool supplies 900 training and 100 validation IDs fixed before training. Normalizers use only the 900 training IDs. All models use full 85×85 input and the shared physical relative-L2 plus derivative objective. Validation selects the epoch. The smooth2 test population was historically viewed, so the experiment does not describe it as an untouched holdout.

Completed implementation evidence:

- `results/interface_preflight.json`: native Full and variable-point interfaces, parameter counts, unit compensation and fixed-input weighted aggregation. Official convolution rejected a real 2,709-point input with a 7,225-point reshape mismatch.
- `results/training_objective_preflight.json`: physical/derivative loss and backward path for all three actual classes.
- `results/request_integration_preflight.json`: initialized full classes exercise Full/Uniform/PI/CPG β=0/1/CA, native/new complete fields, cached-map behavior, and each backbone's own two-block head backward path. This receipt is not trained-accuracy evidence.
- `results/timing_persistence_preflight.json`: completed direct observations survive a profiling interruption and can resume profiling without rerunning direct repetitions. The CPU orchestration check is not a latency benchmark.
- `results/head_normalizer_identity_validation.json`: the actual current normalizer identity passes when declared consistently by backbone/head metadata; mismatched head or backbone identities are rejected. These are identity-gate fixtures, not completed trained heads.
- `results/full_input_path_validation.json`: all three actual classes produce identical CPU Full outputs using original input tensors versus full-range indexed copies. Full now uses the original tensors; this is not a GPU speed result.

The accepted reconstruction is v1.1, with original FP32 coordinates promoted to FP64, unchanged native linear-plus-nearest rules, and default profiling disabled in the main request. New checkpoints must still pass their own validation and every old/new test comparison. CPG/CA geometric patches are rebuilt inside every timed request; interpolation maps are reusable only in the explicitly separated C scene.

The baseline phase contains 7,400 requests per point model and is exported before CA head training. Its own frozen backbone then supplies the head feature/target bank, followed by 1,800 CA requests. The total planned accuracy matrix is 18,600 physical requests / 37,200 old-new metric rows. The timing matrix is 333 groups / 1,323 conditions / 132,300 formal observations, with all comparisons on one admitted GPU. These are expected counts, not completed-result counts.

`protocol.md`, `COMMANDS.md`, `LINEARNO_ADAPTER.md`, `config/dependencies.json`, `config/implementation_manifest.json` and `code/patches/` provide the reviewable protocol and implementation. After real evaluations complete, the persistent postprocess queue produces the result CSV files and replaces this status report with evidence-derived summaries. Parent-owned cross-dataset aggregation provides the independent rAUEC check and final figures. No remote push or historical result replacement has occurred.
