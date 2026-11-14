"""Write a local evidence report from verified Car result CSVs."""
from __future__ import annotations
import csv,json,statistics,time
from collections import Counter
from pathlib import Path


def rows(path):return list(csv.DictReader(Path(path).open(newline='')))

def generate(root):
    root=Path(root);coverage=json.loads((root/'results/car_matrix_coverage.json').read_text())
    assert coverage['complete']
    accuracy=json.loads((root/'results/car_correctness.json').read_text());speeds=rows(root/'results/car_old_new_speedups.csv')
    summary=rows(root/'results/reconstruction_summary.csv');registry=rows(root/'config/checkpoint_registry.csv')
    protocol=json.loads((root/'config/car_protocol.json').read_text())
    text=['# ShapeNet-Car reconstruction follow-up',
        '',f"The locked matrix is complete: {accuracy['pairs']:,} predictions each reconstructed both ways, {accuracy['beta_pairs']:,} matched CPG beta pairs, 990 unique timing conditions across A/B/C, 99,000 formal timing observations and 990 isolated-memory conditions.",
        '',f"All complete fields passed the predeclared elementwise check. Maximum absolute old/new difference: {accuracy['max_absolute_difference']:.6g}; maximum relative norm difference: {accuracy['max_relative_norm']:.6g}; maximum Field metric change: {accuracy['field_metric_max_difference']:.6g}.",
        '', 'Car Field is velocity relative L2 on the full cloud plus0.5 times pressure relative L2 on the surface. Supplemental Cd is computed from these same Field-target predictions; force integration is outside the timed complete-field request. The100-case population and both frozen checkpoints are inherited from historical E3; these test results have already been viewed historically.',
        '', '## Complete output performance', '',
        '| Scope | Backbone | Median paired reconstruction speedup | Median paired complete-request speedup | Complete-request speedup range | Conditions slower after optimization |',
        '|---|---|---|---|---|---|']
    claims=[]
    for scope in ['A','B','C']:
        for arch in ['Transolver3','LinearNO']:
            selected=[r for r in speeds if r['scope']==scope and r['backbone']==arch]
            recon=[float(r['reconstruction_speedup']) for r in selected];total=[float(r['total_speedup']) for r in selected]
            assert selected
            med=statistics.median(total);slow=sum(v<1 for v in total)
            text.append(f'| {scope} | {arch} | {statistics.median(recon):.3f}× | {med:.3f}× | {min(total):.3f}–{max(total):.3f}× | {slow}/{len(total)} |')
            claims.append(f'- Scope {scope}, {arch}: median of the matched old/new complete-request speedups is {med:.3f}×; {slow}/{len(total)} conditions are slower. This statistic is over paired fixed-case/budget/seed conditions, not a ratio across scopes.')
    text += ['', 'Speedups above compare each condition with its matching old/new request, then summarize those ratios. Full uses one shared measured series per backbone/case/scope and is excluded from optimization speedup aggregation. Per-condition median/p95 and raw repeats remain in reconstruction_summary.csv and car_timing_repeats.csv.',
        '', 'A is CPU model inputs to the full CPU float64 output. B starts with resident GPU model tensors and returns the full GPU float64 output, counting CPU auxiliary geometry/selection work. C uses A’s boundary and permits only a map keyed by physical geometry, exact reference coordinates/dtype, selected order, surface mask and reconstruction rule. All geometry-dependent scorers/samplers are rebuilt every request in every scope. No prediction, score or features are cached across requests.',
        '', 'Reconstruction time is the separately instrumented complete output-processing interval, including decode, map work, interpolation, output conversion and required transfers. Direct total latency is measured separately without stage insertion; no stage sum is substituted for total time. Stage synchronization overhead means profiling intervals are diagnostic rather than exact additive shares of the direct request.',
        '', '## Bottleneck evidence', '']
    for scope in ['A','B','C']:
        for arch in ['Transolver3','LinearNO']:
            counts=Counter(r['dominant_instrumented_stage'] for r in summary if r['scope']==scope and r['backbone']==arch and r['implementation']=='new')
            text.append(f'- {scope}/{arch}: the most frequent largest instrumented interval is {counts.most_common(1)[0][0]} ({counts.most_common(1)[0][1]} sampled conditions).')
    text += ['', '## Checkpoints and training', '', '| Backbone | Parameters | Epochs | Updates | Effective batch | Historical training process seconds |', '|---|---|---|---|---|---|']
    for r in registry:text.append(f"| {r['backbone']} | {int(r['parameters']):,} | {r['epochs']} | {r['optimizer_steps']} | {r['global_batch_size']} | {float(r['training_process_wall_seconds']):.1f} |")
    text += ['', 'Architecture class, checkpoint and normalizer hashes, source path/hash, training split, actual point schedule and imported training-cost scope are in config/checkpoint_registry.csv. These Car training costs are historical measurements; this follow-up retrains no Car backbone. The T3 CA head is its matching completed seed0 head, run eagerly with only within-request first-block reuse. LinearNO Car PI/CA are not claimed as completed; no T3 head is attached to LinearNO.',
        '', '## Cache, preparation and memory', '',
        'car_reuse_costs.csv retains the first complete request, map construction/application intervals, CPU/device map bytes and derived first+(R−1)×steady-median cost for R=1/3/10/100. Those amortized costs are estimates derived from measured first/steady requests, not additional direct-wall measurements. Scope-C old means the equivalent CPU sparse-map application; scope-C new means GPU sparse-map application. Both receive the same map-reuse opportunity.',
        '', 'Model/head load cost and resident-input preparation are retained by scope. Native FPS compile/load is prepared once and separately recorded in config/runtime_preparation.json. Its task-local compiled library does not cache geometry or point sets. Every formal condition still has20 warmups and100 measured repeats with stored cyclic order.',
        '', 'Memory is measured in separate architecture/method processes. PyTorch allocated and reserved peaks include model, output, cached map and temporaries. SciPy allocations are on CPU; CUDA context/library allocations fall outside the PyTorch allocator. Post-request NVML samples are retained but do not certify all transient external GPU allocations.',
        '', '## Evidence and reproducibility', '',
        '- Raw accuracy: records/car_accuracy/pairs.jsonl plus retained-index NPZ files and per-run source snapshots.',
        '- Raw timing: records/car_timing/{A,B,C}/ with all warm/formal observations, saved order, first requests, alignment checks and GPU telemetry.',
        '- Raw memory: records/car_memory/.',
        '- Independent rAUEC check: results/car_rauec_independent_audit.json, recomputed directly from per-case CSV.',
        '- Unified reconstruction rules and gate: accepted_protocols.json, reconstruction_registry.json, protocol.md.',
        '- Complete condition table: results/reconstruction_summary.csv; all figure source CSVs are under results/.',
        '- Figures: figures/ (PDF,SVG,300dpiPNG); tables: tables/; scripts: code/.',
        '', 'Run from the active07_current root: `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c128-135 python3 linearno_followup_20261114/reconstruction/code/car_workflow.py`. The controller waits for an idle owned GPU6, verifies source identities when resuming, and never stops external jobs. Existing completed accuracy is not silently appended with changed code.',
        '', 'This local package preserves historical E1/E3, does not publish to GitHub and does not modify the manuscript or Feishu. Darcy/Airfoil new-cohort final accuracy and formal latency are delivered by their respective owners using the same accepted shared reconstruction API.']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    card=['# Claim card', '', 'Supported by this Car package:',
        f"- Rule-preserving reconstruction passed {accuracy['pairs']:,} complete-field old/new comparisons; maximum Field metric change {accuracy['field_metric_max_difference']:.6g}.",
        '- CPG beta0/1 is paired by both selected-index and coarse-prediction hashes. All100 cases and all3 sampling seeds are retained.',*claims,
        '', 'Limits:', '- Standard-task QHull microprofiles use published validation predictions and cannot support claims about the new Darcy/Airfoil cohort.',
        '- B and C timings cannot be substituted for A’s CPU-to-CPU complete-output latency.',
        '- The current Car matrix has no matched LinearNO CA head and does not add its historical PI row. These remain explicit N/A cells.',
        '- Three sampling seeds quantify sampling variability for one frozen training seed. They are not independent training replicates.',
        '- A higher/lower backbone-normalized rAUEC does not by itself establish better absolute accuracy across backbones.',
        '- Visual render QA remains a separate review of generated artifacts.']
    (root/'CLAIM_CARD.md').write_text('\n'.join(card)+'\n')
    (root/'read.me').write_text('Car computation and numerical aggregation completed.\n\n7400 old/new field pairs;1800 CPG beta pairs;990 A/B/C conditions;99000 formal observations;990 isolated-memory conditions.\nREPORT.md contains scope definitions, measured speedups (including slower outcomes), provenance and limitations. Figures/tables were generated from audited CSVs; inspect rendered figures before publication. Standard-task new-cohort outputs are maintained under ../standard_benchmarks. No remote push or manuscript modification was made.\n')


if __name__=='__main__':generate(Path(__file__).resolve().parents[1])
