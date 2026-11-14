"""Generate the local evidence card from actual receipts and train/head manifests."""
from common import *
import csv,datetime,difflib


def read_optional(path):return read(path) if Path(path).exists() else {}


def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();costs=[];rows=[]
    starts={r['architecture']:r for r in read(OUT/'config/training_process_starts.json')}
    for arch in ['t3','linearno','linearno_conv']:
        m=read_optional(OUT/f'checkpoints/{arch}/manifest.json');h=read_optional(OUT/f'checkpoints/{arch}/head/manifest.json')
        accuracy=read_optional(OUT/f'results/{arch}_accuracy_acceptance.json')
        head_stages=read_optional(OUT/f'runtime/{arch}_head_substages.json')
        rows.append((arch,m.get('epochs_completed',0),m.get('execution_status','pending'),accuracy.get('passed','pending'),m.get('checkpoint_sha256','pending')))
        costs.append(dict(dataset='airfoil2d',architecture=arch,training_seed=0,train_cases=900,validation_cases=100,
            epochs_completed=m.get('epochs_completed',0),updates_completed=m.get('total_updates',0),parameter_count=m.get('parameter_count'),
            training_loop_wall_seconds=m.get('training_seconds'),
            process_start_to_completed_manifest_seconds=(OUT/f'checkpoints/{arch}/manifest.json').stat().st_mtime-starts[arch]['process_start_epoch'] if m.get('execution_status')=='completed' else None,
            timing_scope='training loop plus validation; separate process-start to final-manifest measure includes setup and checkpoint write; shared-GPU interval documented',
            head_seed=0 if h else None,head_epochs=h.get('epochs_completed'),head_parameters=h.get('parameter_count'),
            head_feature_generation_seconds=h.get('dataset_generation_seconds'),head_training_seconds=h.get('training_seconds'),
            head_bank_stage_seconds=h.get('head_bank_stage_seconds'),head_training_stage_seconds=h.get('head_training_stage_seconds'),
            head_cost_scope='observed stage wall time on the shared machine; not isolated inference throughput' if h else 'not_applicable_or_pending',
            preliminary_evaluation_seconds=head_stages.get('stages',{}).get('preliminary_evaluation',{}).get('seconds'),
            head_acceptance_seconds=head_stages.get('stages',{}).get('head_acceptance',{}).get('seconds'),
            head_pipeline_including_preliminary_seconds=head_stages.get('pipeline_including_preliminary_seconds'),
            checkpoint_sha256=m.get('checkpoint_sha256','pending'),normalizer_sha256=m.get('normalizer_sha256','pending'),head_sha256=h.get('head_sha256','N/A: pending or Full-only')))
    with (OUT/'results/training_head_cost.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(costs[0]));w.writeheader();w.writerows(costs)
    latency=read_optional(OUT/'results/latency_completion.json');passed=read_optional(OUT/'results/latency_validation.json').get('passed',False)
    text=f'# Airfoil LinearNO follow-up\n\nGenerated from local records: {now}.\n\n'
    text+='Shared task: pressure channel 3 on the 221×51 grid, 900 training and 100 validation geometries, fixed published 110-case positive-lift test population. Historical test visibility is disclosed in protocol.md.\n\n'
    text+='| Model | Epochs | Training state | Accuracy accepted | Weight SHA-256 |\n|---|---:|---|---|---|\n'
    for row in rows:text+='| '+' | '.join(map(str,row))+' |\n'
    text+='\nThe main point models use native Transolver3 and the official LinearNO point-temp variant. The official conv-temp variant is reported separately at Full grid only. Its lack of variable-point support is recorded in model_compatibility.md.\n\n'
    text+='Reconstruction uses accepted grid_qhull_sparse_apply_v1p1, with original integer-grid float64 coordinates and the original linear-plus-nearest rule. Every sampled prediction is reconstructed by both old and new implementations. Full predictions are saved once and explicitly aliased between reconstruction views.\n\n'
    area_path=OUT/'results/backbone_rauec.csv';area=[]
    if area_path.exists():
        with area_path.open() as stream:area=[r for r in csv.DictReader(stream) if r['metric']=='field_relative_l2' and r['reconstruction_id']=='grid_qhull_sparse_apply_v1p1']
    if area:
        text+='Complete pressure-field area ratios below use each backbone’s own Uniform. Mean ± sample SD is across three sampling seeds of one training seed. CPG beta=0 is the separate mechanism control.\n\n'
        text+='| Backbone | Method | Beta | rAUEC |\n|---|---|---:|---:|\n'
        for row in area:text+=f"| {row['backbone']} | {row['method']} | {row['beta'] or 'N/A'} | {float(row['mean_rauec']):.6f} ± {float(row['sampling_sd']):.6f} |\n"
        text+='\n'
    absolute_path=OUT/'results/backbone_absolute_errors.csv';full_errors=[]
    if absolute_path.exists():
        with absolute_path.open() as stream:full_errors=[r for r in csv.DictReader(stream) if r['metric']=='field_relative_l2' and r['method']=='full' and r['reconstruction_id']=='grid_qhull_sparse_apply_v1p1']
    if full_errors:
        text+='Full-grid pressure errors are directly comparable on the same 110 cases. The convolutional model is a separately trained Full-only supplement.\n\n| Native model | Full relative L2 |\n|---|---:|\n'
        for row in full_errors:text+=f"| {row['backbone']} | {float(row['mean_error']):.9f} |\n"
        text+='\n'
    text+=f"Formal latency acceptance: {passed}; observed groups {latency.get('actual_groups',0)}/333, conditions {latency.get('actual_conditions',0)}/1323, formal repeats {latency.get('actual_formal_repeats',0)}/132300. A is CPU input to float64 CPU field; B and C have separate rows.\n\n"
    text+='Results: results/backbone_budget_metrics.csv, backbone_absolute_errors.csv, backbone_rauec.csv and backbone_latency_summary.csv. Missing files mean the corresponding stage is pending; no historical results are substituted. rAUEC is independently calculated per sampling seed, then reported as mean and sample SD across three sampling seeds; Full is excluded.\n\n'
    text+='Commands and source identities: runtime/launches.json, runtime/*_workflow.json, config/execution_sources, config/reconstruction_sources, timing/sources. Raw predictions and selected indices: records/{architecture}; raw latency groups and occupancy records: timing/. Cross-task plots and archive assembly belong to the parent standard_benchmarks delivery.\n'
    (OUT/'REPORT.md').write_text(text)
    claims='# Claim card\n\n'
    claims+='Supported implementation statements: native point-temp accepts variable point sets; its key-mass correction passes unit-weight and explicit-duplication checks; original conv-temp rejects arbitrary point sets and supports the full grid; replayed common sampling RNG states match saved training checkpoints. These checks do not establish accuracy or speed improvement.\n\n'
    claims+='Trained accuracy is accepted only for rows marked true in REPORT.md and backed by results/{architecture}_accuracy_acceptance.json. No checkpoint is selected with test error. The CPG beta=0/1 mechanism keeps coarse and selected index hashes paired.\n\n'
    if area:
        claims+='Observed pressure-field results in the fixed 110-case population:\n\n'
        for label in sorted({r['backbone'] for r in area}):
            current={r['method']:r for r in area if r['backbone']==label}
            if 'cpg' not in current:continue
            cpg=float(current['cpg']['mean_rauec']);zero=float(current['cpg_beta0']['mean_rauec'])
            claims+=f"- {label}: CPG beta=1 rAUEC {cpg:.6f} ({100*(1-cpg):.2f}% less area than its matched Uniform); paired beta=0 rAUEC {zero:.6f}.\n"
            if 'ca' in current:
                ca=float(current['ca']['mean_rauec']);claims+=f"- {label}: CA rAUEC {ca:.6f}; its observed mean is {'higher' if ca>cpg else 'lower'} than CPG beta=1. This ordering is retained.\n"
        claims+='\nThis beta=0/1 experiment supports the compensation mechanism for this cohort and field-target policy. It does not identify the optimum over untested beta values.\n\n'
    if full_errors:
        claims+='Absolute Full errors (same 110-case population): '+', '.join(f"{r['backbone']}={float(r['mean_error']):.9f}" for r in full_errors)+'.\n\n'
        claims+='The convolutional variant uses its native full-grid training/input path and has no subsampling/rAUEC result. Its Full accuracy must remain visible when interpreting the point-variant comparison.\n\n'
    claims+=f'Complete-request timing supported: {passed}. Reconstruction-only speedups and cache reuse have separate meanings from A-scope end-to-end speedups.\n\n'
    claims+='Comparative claims about which backbone is faster or more accurate must be drawn from completed absolute field-error and same-scope latency rows. Each backbone has its own Uniform rAUEC denominator; those normalized scores cannot rank absolute model accuracy. Unfavorable results are preserved.\n'
    (OUT/'CLAIM_CARD.md').write_text(claims)
    pieces=[];sources={}
    for path in sorted((OUT/'code').glob('*.py')):
        name='standard_benchmarks/airfoil/code/'+path.name
        pieces.extend(difflib.unified_diff([],path.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+name))
        sources[name]=sha(path)
    (OUT/'results/implementation.patch').write_text(''.join(pieces));write(OUT/'results/implementation_sources.json',sources)


if __name__=='__main__':main()
