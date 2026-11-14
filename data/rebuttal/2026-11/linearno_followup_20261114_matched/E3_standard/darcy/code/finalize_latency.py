"""Traceable condition-level Darcy timing, profiling, memory and reuse tables."""
from __future__ import annotations
import argparse,csv,collections,json
import numpy as np
from common import OUT,read,write,digest


KEYS=['dataset','backbone','case_id','method','budget_fraction','sampling_seed','implementation','scope']


def key(row):return tuple(str(row.get(k)) for k in KEYS)


def csv_write(path,rows):
    if not rows:raise ValueError(f'No observations for {path}')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    directory=OUT/'timing';destination=OUT/'results';groups=[]
    for path in sorted((directory/'groups').glob('*.json')):
        if path.name=='heartbeat.json':continue
        value=read(path)
        if value.get('execution_status')=='completed':groups.append((path,value))
    memories={};memory_inputs={}
    for path in sorted((directory/'memory').glob('*.json')):
        if path.name.endswith('_gpu.json'):continue
        value=read(path)
        if value.get('execution_status')!='completed':continue
        memory_inputs[str(path)]=digest(path)
        for row in value['rows']:
            identity=key(row)
            if identity in memories:raise ValueError('duplicate memory condition')
            memories[identity]=row
    rows=[];raw=[];profile=[];reuse=[];inputs={}
    for path,group in groups:
        if group.get('profile_execution_status')!='completed' and not args.allow_partial:
            raise RuntimeError(f'Profile incomplete: {path}')
        inputs[str(path)]=digest(path)
        first={r['condition']:r for r in group['first_requests']}
        summaries={r['condition']:r for r in group['summary']}
        for condition,meta in group['metadata'].items():
            formal=[r for r in group['raw'] if r['condition']==condition and r['kind']=='formal']
            warmup=[r for r in group['raw'] if r['condition']==condition and r['kind']=='warmup']
            assert len(formal)==100 and len(warmup)==20
            assert {r['repeat_id'] for r in formal}==set(range(100))
            timings=np.array([r['total_ms'] for r in formal]);assert np.isfinite(timings).all() and (timings>0).all()
            for r in formal+warmup:raw.append(dict(**meta,group_id=group['group_id'],condition=condition,
                kind=r['kind'],repeat_id=r['repeat_id'],order_position=r['order_position'],total_ms=r['total_ms'],
                order_seed=group['order']['seed'],source_json=str(path)))
            components=group.get('profiles',{}).get(condition,[])
            for r in components:profile.append(dict(**meta,group_id=group['group_id'],condition=condition,**r))
            reconstruction=[r['ms'] for r in components if r['component']=='reconstruction_complete_output']
            if meta['method']!='full' and len(reconstruction)!=5 and not args.allow_partial:raise ValueError('incomplete reconstruction profile')
            sample=first[condition];check=sample['output_check'];memory=memories.get(key(meta))
            if memory is None and not args.allow_partial:raise ValueError(f'memory absent: {key(meta)}')
            row=dict(**meta,group_id=group['group_id'],condition=condition,
                direct_total_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),
                mean_ms=float(timings.mean()),sample_sd_ms=float(timings.std(ddof=1)),formal_repeats=100,warmups=20,
                reconstruction_ms=0. if meta['method']=='full' else float(np.median(reconstruction)) if reconstruction else None,
                reconstruction_timing='separate 5-pass synchronized full reconstruction incl required transfers; not a direct-total decomposition',
                first_request_ms=sample['first_request_ms'],field_error=check['field_error'],
                selected_indices_hash=check['selected_indices_hash'],coarse_indices_hash=check['coarse_indices_hash'],
                compensation_weights_hash=check['compensation_weights_hash'],
                full_field_max_absolute_difference=check['full_field_max_absolute_difference'],
                full_field_relative_difference=check['full_field_relative_difference'],
                accepted_prediction_arrays=check['accepted_prediction_arrays'],
                cached_map_cpu_bytes=check.get('cached_map_cpu_bytes'),cached_map_gpu_bytes=check.get('cached_map_gpu_bytes'),
                reuse_assumption='same physical case, coordinates and selected order; recompute model/score/selection each call' if meta['scope'].startswith('C_') else 'no interpolation-map cache',
                memory_status='measured' if memory else 'N/A: memory stage incomplete',source_json=str(path),source_sha256=inputs[str(path)])
            if memory:
                for name in ['peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','first_request_peak_cuda_allocated_bytes','first_request_peak_cuda_reserved_bytes','allocator_scope','external_scope']:
                    row[name]=memory[name]
                samples=[m['device_used_MiB'] for m in memory['nvml_post_samples'] if m and 'device_used_MiB' in m]
                row['nvml_post_max_device_MiB']=max(samples) if samples else None
            rows.append(row)
            if meta['scope'].startswith('C_') and meta['method']!='full':
                cold=[r['ms'] for r in components if r['component']=='first_cache_map_build']
                for count in [1,2,5,10,20,50,100]:
                    # A derived sum of observed requests, not a contiguous n-request lifetime measurement.
                    seq=sample['first_request_ms']+sum(r['total_ms'] for r in sorted(formal,key=lambda r:r['repeat_id'])[:count-1])
                    reuse.append(dict(**meta,requests=count,assembled_observation_sum_ms=seq,
                        modeled_median_total_ms=sample['first_request_ms']+(count-1)*row['direct_total_ms'],
                        first_request_ms=sample['first_request_ms'],steady_median_ms=row['direct_total_ms'],
                        separate_map_build_ms=cold[0] if cold else None,
                        cost_definition='assembled first request + n-1 post-warmup observations; excludes 20 warmups; not a separately measured contiguous request sequence',
                        model_definition='first request + (n-1)*formal median, derived estimate not separate measurement',
                        cached_map_cpu_bytes=row['cached_map_cpu_bytes'],cached_map_gpu_bytes=row['cached_map_gpu_bytes'],
                        reuse_assumption=row['reuse_assumption'],source_json=str(path)))
    identities=[key(r) for r in rows];assert len(set(identities))==len(identities)
    complete=len(groups)==333 and len(rows)==1323 and len(memories)==1323 and all(g.get('profile_execution_status')=='completed' for _,g in groups)
    if not args.allow_partial and not complete:raise ValueError(f'matrix incomplete: groups={len(groups)}, timing={len(rows)}, memory={len(memories)}')
    if rows:
        csv_write(destination/'backbone_latency_summary.csv',rows);csv_write(destination/'latency_raw_repeats.csv',raw)
        csv_write(destination/'latency_profiles.csv',profile);csv_write(destination/'latency_reuse_costs.csv',reuse)
    write(destination/'latency_completion.json',dict(execution_status='completed' if complete else 'partial',
        expected_groups=333,actual_groups=len(groups),expected_conditions=1323,actual_conditions=len(rows),
        expected_formal_repeats=132300,actual_formal_repeats=sum(r['kind']=='formal' for r in raw),
        expected_memory_conditions=1323,actual_memory_conditions=len(memories),
        group_inputs=inputs,memory_inputs=memory_inputs,
        full_alias='one actual physical Full timing series per backbone/case/scope; old/new reference it',
        C_scope='separate cached-map scene; never divided by scope A Full',
        external_memory_scope='PyTorch allocator peak plus post-request NVML samples; transient external GPU peaks and CPU allocations not certified'))
    print(json.dumps(dict(complete=complete,groups=len(groups),conditions=len(rows),memory=len(memories))))


if __name__=='__main__':main()
