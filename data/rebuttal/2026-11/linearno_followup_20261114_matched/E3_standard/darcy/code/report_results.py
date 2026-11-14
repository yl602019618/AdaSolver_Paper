"""Build the local evidence report only from complete, traceable result tables."""
from __future__ import annotations
import csv,collections,json
from common import OUT,read,write,digest
from finalize_latency import csv_write


def load(name):return list(csv.DictReader((OUT/'results'/name).open()))


def main():
    accuracy=read(OUT/'results/accuracy_completion.json');latency=read(OUT/'results/latency_completion.json')
    if accuracy['execution_status']!='completed' or latency['execution_status']!='completed':raise RuntimeError('local result matrices incomplete')
    times=load('backbone_latency_summary.csv');rauec=load('backbone_rauec.csv');absolute=load('backbone_absolute_errors.csv')
    groups=collections.defaultdict(dict)
    for row in times:
        if row['method']=='full':continue
        groups[tuple(row[k] for k in ['backbone','case_id','method','budget_fraction','sampling_seed','scope'])][row['implementation']]=row
    comparisons=[]
    for key,group in groups.items():
        if set(group)!=set(['old','new']):raise ValueError('old/new latency pair absent')
        old,new=group['old'],group['new']
        assert all(old[k]==new[k] for k in ['checkpoint_id','normalizer_id','selected_indices_hash','output_dtype','output_device'])
        comparisons.append(dict(dataset='darcy',backbone=key[0],case_id=key[1],method=key[2],budget_fraction=key[3],sampling_seed=key[4],scope=key[5],
            checkpoint_id=old['checkpoint_id'],old_reconstruction_id=old['reconstruction_id'],new_reconstruction_id=new['reconstruction_id'],
            old_field_error=old['field_error'],new_field_error=new['field_error'],
            old_reconstruction_ms=old['reconstruction_ms'],new_reconstruction_ms=new['reconstruction_ms'],
            reconstruction_speedup=float(old['reconstruction_ms'])/float(new['reconstruction_ms']),
            old_direct_total_ms=old['direct_total_ms'],new_direct_total_ms=new['direct_total_ms'],
            complete_request_speedup=float(old['direct_total_ms'])/float(new['direct_total_ms']),
            old_p95_ms=old['p95_ms'],new_p95_ms=new['p95_ms'],
            old_peak_allocated_bytes=old['peak_cuda_allocated_bytes'],new_peak_allocated_bytes=new['peak_cuda_allocated_bytes'],
            old_peak_reserved_bytes=old['peak_cuda_reserved_bytes'],new_peak_reserved_bytes=new['peak_cuda_reserved_bytes'],
            preparation_old_first_request_ms=old['first_request_ms'],preparation_new_first_request_ms=new['first_request_ms'],
            output_dtype=old['output_dtype'],output_device=old['output_device'],reuse_assumption=old['reuse_assumption'],
            old_source=old['source_json'],new_source=new['source_json']))
    csv_write(OUT/'results/reconstruction_summary.csv',comparisons)
    registry=load('checkpoint_registry.csv')
    for row in registry:row['latency_acceptance']='completed';row['independent_rauec_review']='parent-owned separate receipt'
    csv_write(OUT/'results/checkpoint_registry.csv',registry)
    lines=['# Darcy matched cohort results','',
        'All local model, accuracy and A/B/C timing stages are complete. Cross-dataset aggregation and independent rAUEC review remain separate receipts.','',
        'Models share the locked 900/100 split, 900-only normalizers, 1,000 epochs / 225,000 updates and physical-field plus derivative training loss. The smooth2 test population was historically viewed. The official convolution variant is Full-only because sparse inputs violate its native 85×85 shape.','',
        f'Physical accuracy requests: {accuracy["physical_requests"]:,}; raw old/new metric rows: {accuracy["total_rows"]:,}. Timing conditions: {latency["actual_conditions"]:,}; formal repeats: {latency["actual_formal_repeats"]:,}.','',
        '| Backbone | Method | Reconstruction | rAUEC mean ± sampling SD (n=3) |','| --- | --- | --- | --- |']
    for row in rauec:
        lines.append(f'| {row["backbone"]} | {row["method"]} | {row["reconstruction_id"]} | {float(row["mean"]):.6f} ± {float(row["sampling_sd"]):.6f} |')
    lines.extend(['','The three budgets are 37.5%, 50% and 75%. Each seed forms an area ratio against its own backbone/checkpoint Uniform. Full is excluded from the integral. These normalized ratios do not compare absolute errors across backbones.','',
        'Absolute errors are in `results/backbone_absolute_errors.csv`; each row includes its same-checkpoint Full error. Formal timing, independent profiling, cold/steady allocator measurements and cache costs are in `results/backbone_latency_summary.csv`, `latency_profiles.csv`, `latency_raw_repeats.csv` and `latency_reuse_costs.csv`.','',
        'The CPU→CPU scene returns float64 fields. GPU-resident inputs return GPU float64 fields in B. C reuses only geometry/selected-index reconstruction maps and recomputes acquisition/predictions. Speed ratios in `reconstruction_summary.csv` always remain within the same scene, model, point set and output contract.','',
        'Peak memory covers PyTorch CUDA allocator peaks plus post-request NVML samples; the samples do not certify transient external-library GPU peaks or CPU SciPy allocations. Model loading and kernel warmup are recorded in `timing/setup.json`. No CUDA Graph or approximate-neighbor path is used.',''])
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    claims=['# Darcy claim card','',
        '| Claim | Evidence and allowed interpretation |','| --- | --- |',
        '| Same reconstruction rule on both point backbones | Checkpoint-specific validation and every test old/new pair passed the locked elementwise and field-metric tolerances. |',
        '| β=1 improves over β=0 | Assess paired CPG records and the rAUEC mechanism rows; no universal best-β claim follows from nominal inverse proposal correction. Retain adverse rows. |',
        '| LinearNO is slower or faster than Transolver3 | Assess same-scene Full and sampled direct request observations. Official point-temp and conv-temp are separate variants. |',
        '| Reconstruction optimization improves latency | Use each old/new paired row. Reconstruction-only and complete-request speedup are separate columns; ratios below 1 are retained. |',
        '| New untouched holdout | Unsupported: historical smooth2 data had previously been viewed. |',
        '| Official conv-temp supports variable-point inference | Unsupported: native reshape rejected sparse input. Its valid Full reference is included. |',
        '| Complete publication-ready cross-task comparison | Pending parent cross-dataset aggregation, independent rAUEC audit and figures. |','']
    (OUT/'CLAIM_CARD.md').write_text('\n'.join(claims))
    (OUT/'read.me').write_text('Darcy local training, CA, accuracy and A/B/C latency matrices complete. REPORT.md and CLAIM_CARD.md bind the local scope; cross-dataset publication artifacts remain parent-owned. No remote push performed.\n')
    write(OUT/'results/local_report_completion.json',dict(execution_status='completed',
        inputs={name:digest(OUT/'results'/name) for name in ['backbone_rauec.csv','backbone_absolute_errors.csv','backbone_latency_summary.csv']},
        report=str(OUT/'REPORT.md'),claims=str(OUT/'CLAIM_CARD.md')))


if __name__=='__main__':main()
