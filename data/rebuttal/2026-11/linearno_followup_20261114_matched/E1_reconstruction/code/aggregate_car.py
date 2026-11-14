"""Audit and aggregate immutable Car pairs/repeats; never fill missing cells.

Default mode requires the entire locked matrix. --partial produces a coverage
receipt only and cannot emit publication tables from incomplete evidence.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics,time
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]
METRICS=('field_error','velocity_rel_l2','pressure_rel_l2','cd_relative_error')
LABELS={'full':'Full','uniform':'Uniform','pi':'PI','cpg_beta0':'CPG beta=0','cpg_beta1':'CPG beta=1','ca':'CA'}
IDENTITY=('backbone','case_id','variant','sampling_seed','budget_fraction')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csvrows(path):return list(csv.DictReader(Path(path).open(newline='')))
def key(row):return tuple(row[k] for k in IDENTITY)
def emit_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def emit_csv(path,rows,fields=None):
    rows=list(rows);fields=fields or list(rows[0] if rows else {})
    assert fields,'Empty files need explicit column headers'
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='raise');w.writeheader()
        for row in rows:w.writerow({k:('N/A' if row.get(k) is None else row.get(k)) for k in fields})


def expected_accuracy(protocol):
    return {(arch,case,variant,seed,budget)
        for arch,variants in protocol['accuracy_variants'].items() for case in protocol['test_ids'] for variant in variants
        for budget in ([1.] if variant=='full' else protocol['budgets'])
        for seed in ([None] if variant=='full' else protocol['sampling_seeds'])}


def coverage(protocol):
    source=OUT/'records/car_accuracy/pairs.jsonl'
    rows=[json.loads(line) for line in source.open()] if source.exists() else []
    keys=[key(r) for r in rows];want=expected_accuracy(protocol)
    assert len(set(keys))==len(keys),'Duplicate accuracy requests'
    assert set(keys)<=want,'Unexpected accuracy requests'
    timing={scope:sum(read(p)['conditions'] for p in (OUT/'records/car_timing'/scope).glob('*/complete.json')
                 if read(p).get('execution_status')=='completed') for scope in ['A','B','C']}
    memory=sum(len(read(p)['rows']) for p in (OUT/'records/car_memory').glob('*.json')
               if read(p).get('execution_status')=='completed' and 'rows' in read(p))
    return dict(accuracy_pairs_actual=len(keys),accuracy_pairs_expected=len(want),beta_pairs_expected=1800,
        unique_timing_conditions_actual=timing,unique_timing_conditions_expected_per_scope=330,
        formal_repeats_per_condition=100,warmups_per_condition=20,memory_conditions_actual=memory,
        memory_conditions_expected=990,complete=len(keys)==len(want) and all(v==330 for v in timing.values()) and memory==990,
        unavailable={'LinearNO Car PI':'Not in completed historical Car E3 policy matrix; not added in this reconstruction follow-up',
                     'LinearNO Car CA':'No completed head matching this LinearNO backbone; Transolver3 head is not borrowed'})


def aggregate_accuracy(protocol):
    raw=OUT/'records/car_accuracy/pairs.jsonl';rows=[json.loads(line) for line in raw.open()]
    assert {key(r) for r in rows}==expected_accuracy(protocol) and len(rows)==7400
    assert read(OUT/'runtime/car_accuracy.json')['execution_status']=='completed'
    index_verified=set();registry={};flat=[];paired={key(r):r for r in rows};beta_pairs=0
    for line_no,row in enumerate(rows,1):
        assert row['difference']['violations']==0
        assert row['metric_target']=='field' and row['training_seed']==0 and row['output_dtype']=='float64'
        arch=row['backbone'];binding=(row['operator_sha256'],row['normalizer_sha256'])
        if arch in registry:assert registry[arch]==binding,'Mixed checkpoints/normalizers'
        registry[arch]=binding
        index_sha=row['selected_indices_sha256']
        if index_sha not in index_verified:
            ids=np.load(OUT/row['indices_file'])['indices'];assert sha_array(ids)==index_sha
            assert len(ids)==row['retained_points'] and len(np.unique(ids))==len(ids)
            index_verified.add(index_sha)
        if row['variant']=='cpg_beta0':
            mate=paired[(arch,row['case_id'],'cpg_beta1',row['sampling_seed'],row['budget_fraction'])]
            assert index_sha==mate['selected_indices_sha256']
            assert row['coarse_prediction_sha256']==mate['coarse_prediction_sha256'];beta_pairs+=1
        for implementation in ['old','new']:
            for metric in METRICS:
                error=row[implementation+'_metrics'][metric];assert math.isfinite(error) and error>=0
                flat.append(dict(dataset='shapenet_car',backbone=arch,variant=row['variant'],method=LABELS[row['variant']],
                    implementation=implementation,reconstruction_id=row[implementation+'_reconstruction_id'],
                    metric=metric,metric_target='field',population='historical E3 all100',case_id=row['case_id'],
                    budget_fraction=row['budget_fraction'],sampling_seed=row['sampling_seed'],training_seed=0,
                    beta=row['beta'],error=error,retained_points=row['retained_points'],
                    checkpoint_sha256=binding[0],normalizer_sha256=binding[1],selected_indices_sha256=index_sha,
                    source_file=str(raw.relative_to(OUT)),source_line=line_no))
    assert beta_pairs==1800
    emit_csv(OUT/'results/car_per_case_metrics.csv',flat)
    groups=defaultdict(list)
    for row in flat:groups[tuple(row[k] for k in ['backbone','implementation','reconstruction_id','variant','metric','budget_fraction','sampling_seed'])].append(row)
    seeds=[]
    for identity,values in sorted(groups.items(),key=str):
        assert len(values)==100 and len({v['case_id'] for v in values})==100
        arch,impl,rid,variant,metric,budget,seed=identity
        seeds.append(dict(dataset='shapenet_car',backbone=arch,implementation=impl,reconstruction_id=rid,variant=variant,
            method=LABELS[variant],metric=metric,budget_fraction=budget,sampling_seed=seed,
            error_mean=math.fsum(v['error'] for v in values)/100,case_count=100,
            retained_points_min=min(v['retained_points'] for v in values),retained_points_max=max(v['retained_points'] for v in values),
            checkpoint_sha256=registry[arch][0],normalizer_sha256=registry[arch][1],population='historical E3 all100',
            metric_target='field',training_seed=0))
    emit_csv(OUT/'results/car_budget_seed_metrics.csv',seeds)
    budget_groups=defaultdict(list)
    for row in seeds:budget_groups[tuple(row[k] for k in ['backbone','implementation','variant','metric','budget_fraction'])].append(row)
    budgets=[]
    for identity,values in sorted(budget_groups.items(),key=str):
        first=values[0];errors=[v['error_mean'] for v in values];n=1 if first['variant']=='full' else 3
        assert len(values)==n
        budgets.append({k:first[k] for k in ['dataset','backbone','implementation','reconstruction_id','variant','method','metric',
            'budget_fraction','checkpoint_sha256','normalizer_sha256','population','metric_target','training_seed']}|
            dict(error_mean=statistics.mean(errors),sampling_sd=statistics.stdev(errors) if n==3 else None,
            sampling_n=3 if n==3 else 0,case_count=100,missing_reason=None if n==3 else 'Full has no sampling seed'))
    emit_csv(OUT/'results/car_budget_metrics.csv',budgets)
    lookup={(r['backbone'],r['implementation'],r['variant'],r['metric'],r['sampling_seed'],r['budget_fraction']):r['error_mean'] for r in seeds}
    ratios=[]
    for arch,variants in protocol['accuracy_variants'].items():
        for impl in ['old','new']:
            for metric in METRICS:
                for variant in [v for v in variants if v!='full']:
                    for seed in protocol['sampling_seeds']:
                        b=protocol['budgets'];curve=[lookup[arch,impl,variant,metric,seed,x] for x in b]
                        uniform=[lookup[arch,impl,'uniform',metric,seed,x] for x in b]
                        area=math.fsum((b[k+1]-b[k])*(curve[k]+curve[k+1])/2 for k in range(2))
                        denominator=float(np.trapezoid(uniform,b));assert denominator>0
                        ratio=area/denominator
                        # Independent summation layout checks the three-point integral.
                        weights=np.array([.0625,.1875,.125]);assert np.isclose(area,np.dot(curve,weights),rtol=1e-14)
                        if variant=='uniform':
                            assert np.isclose(ratio,1.,rtol=1e-14);ratio=1.
                        ratios.append(dict(dataset='shapenet_car',backbone=arch,implementation=impl,variant=variant,method=LABELS[variant],
                            metric=metric,sampling_seed=seed,AUEC=area,uniform_AUEC=denominator,rAUEC=ratio,
                            checkpoint_sha256=registry[arch][0],normalizer_sha256=registry[arch][1],
                            reconstruction_id=protocol[impl+'_reconstruction'],population='historical E3 all100',
                            metric_target='field',training_seed=0,budgets='0.375|0.5|0.75'))
    emit_csv(OUT/'results/car_rauec_seed.csv',ratios)
    rg=defaultdict(list)
    for row in ratios:rg[tuple(row[k] for k in ['backbone','implementation','variant','metric'])].append(row)
    rauec=[]
    for identity,values in sorted(rg.items(),key=str):
        assert len(values)==3
        first=values[0];x=[v['rAUEC'] for v in values]
        rauec.append({k:first[k] for k in ['dataset','backbone','implementation','reconstruction_id','variant','method','metric',
            'checkpoint_sha256','normalizer_sha256','population','metric_target','training_seed','budgets']}|
            dict(rAUEC_mean=statistics.mean(x),rAUEC_sampling_sd=statistics.stdev(x),sampling_n=3))
    emit_csv(OUT/'results/car_rauec.csv',rauec)
    # Absolute error table retains Full separately from the integrated interval.
    emit_csv(OUT/'results/car_absolute_errors.csv',[r for r in budgets if r['metric']=='field_error'])
    correctness=dict(status='passed',pairs=len(rows),beta_pairs=beta_pairs,unique_index_files=len(index_verified),
        max_absolute_difference=max(r['difference']['max_abs'] for r in rows),
        max_relative_norm=max(r['difference']['relative_norm'] for r in rows),
        field_metric_max_difference=max(r['metric_differences']['field_error'] for r in rows),
        max_metric_difference={metric:max(r['metric_differences'][metric] for r in rows) for metric in METRICS},
        raw_file=str(raw.relative_to(OUT)),raw_sha256=sha(raw),
        rAUEC_definition='Per-seed trapezoidal area of mean100-case errors / matched backbone Uniform area; mean and sample SD(ddof=1) across3 seeds; noFull point')
    emit_json(OUT/'results/car_correctness.json',correctness)
    return paired,budgets,rauec,correctness


def sha_array(ids):return hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()
def fnum(value):return None if value is None else float(value)


def aggregate_timing(protocol,accuracy):
    memory={}
    for file in (OUT/'records/car_memory').glob('*.json'):
        doc=read(file)
        if doc.get('execution_status')!='completed' or 'rows' not in doc:continue
        for row in doc['rows']:
            k=(row['scope'],row['backbone'],row['variant'],row['implementation'],row['case_id'],row['budget_fraction'],row['sampling_seed'])
            assert k not in memory;memory[k]={**row,'source_hashes':doc['source_hashes']}
    assert len(memory)==990
    results=[];all_repeats=[];all_profiles=[];reuse=[]
    for scope in ['A','B','C']:
        count=0
        for path in sorted((OUT/'records/car_timing'/scope).glob('*/complete.json')):
            doc=read(path);assert doc['execution_status']=='completed';directory=path.parent
            raw=csvrows(directory/'timing_repeats.csv');summaries=csvrows(directory/'summary.csv');profiles=read(directory/'profiles.json')
            caches=read(directory/'cache.json');firsts={r['condition']:r for r in read(directory/'first_requests.json')}
            preparation=read(directory/'preparation.json');order=read(directory/'order.json')
            assert read(directory/'alignment.json')['passed'];base=order['base'];assert len(base)==doc['conditions']
            for row in raw:
                rep=int(row['repeat_id']);assert base[(rep+int(row['order_position']))%len(base)]==row['condition']
                value=float(row['total_ms']);assert math.isfinite(value) and value>0
                all_repeats.append(dict(dataset='shapenet_car',case_id=doc['case_id'],budget_fraction=doc['budget_fraction'],
                    sampling_seed=doc['sampling_seed'],source_file=str((directory/'timing_repeats.csv').relative_to(OUT)),**row))
            for row in profiles:all_profiles.append(dict(case_id=doc['case_id'],budget_fraction=doc['budget_fraction'],sampling_seed=doc['sampling_seed'],timing_scope=scope,**row))
            for summary in summaries:
                condition=summary['condition'];arch,variant,impl=condition.split(':');count+=1
                r=[r for r in raw if r['condition']==condition]
                assert {int(x['repeat_id']) for x in r if x['kind']=='warmup'}==set(range(20))
                formal=[float(x['total_ms']) for x in r if x['kind']=='formal'];assert len(formal)==100
                assert {int(x['repeat_id']) for x in r if x['kind']=='formal'}==set(range(100))
                direct=float(np.median(formal));p95=float(np.percentile(formal,95))
                assert np.isclose(direct,float(summary['median_ms']),rtol=1e-14)
                assert np.isclose(p95,float(summary['p95_ms']),rtol=1e-14)
                active=[r for r in profiles if r['condition']==condition and (scope!='C' or variant=='full' or r['cache_hit'])]
                component=defaultdict(list)
                for row in active:component[row['component']].append(row['ms'])
                assert 'reconstruction_pipeline_ms' in component
                medians={k:float(np.median(v)) for k,v in component.items()}
                top=['coarse_index_ms','sampling_geometry_prepare_ms','coarse_forward_io_decode_ms','coarse_head_io_decode_ms',
                     'score_ms','selection_compensation_ms','final_forward_input_io_ms','reconstruction_pipeline_ms']
                dominant=max((k for k in top if k in medians),key=lambda k:medians[k])
                a=accuracy[(arch,doc['case_id'],variant,doc['sampling_seed'],doc['budget_fraction'])]
                mem=memory[(scope,arch,variant,impl,doc['case_id'],doc['budget_fraction'],doc['sampling_seed'])]
                assert mem['source_hashes']['car_request.py']==doc['source_hashes']['car_request.py']
                assert mem['source_hashes']['reconstruct_api.py']==doc['source_hashes']['reconstruct_api.py']
                maps=[r for r in caches if r['condition']==condition]
                prep=next(r for r in preparation if r['backbone']==arch and r['variant']==variant)
                rid='full_direct_output' if variant=='full' else protocol[('old' if impl=='old' else 'new')+'_reconstruction']
                if scope=='C' and variant!='full':rid=protocol['new_reconstruction']
                result=dict(dataset='shapenet_car',backbone=arch,variant=variant,method=LABELS[variant],implementation=impl,
                    budget_fraction=doc['budget_fraction'],sampling_seed=doc['sampling_seed'],case_id=doc['case_id'],
                    reconstruction_id=rid,scope=scope,output_dtype='float64',output_device='GPU' if scope=='B' else 'CPU',
                    input_device='GPU model tensors + CPU geometry metadata' if scope=='B' else 'CPU',
                    old_new_field_error=a['metric_differences']['field_error'],old_new_field_max_abs=a['difference']['max_abs'],
                    reconstruction_ms=medians['reconstruction_pipeline_ms'],direct_total_ms=direct,p95_ms=p95,
                    formal_repeats=100,warmups=20,peak_memory_allocated_bytes=mem['peak_cuda_allocated_bytes'],
                    peak_memory_reserved_bytes=mem['peak_cuda_reserved_bytes'],
                    external_memory_scope=mem['external_scope'],nvml_post_max_bytes=max((v for v in mem['nvml_post_samples'] if v is not None),default=None),
                    preparation_cost_ms=prep['geometry_preparation_ms']+prep['resident_input_h2d_ms'],
                    first_request_ms=firsts[condition]['first_request_ms'],
                    reuse_assumption='same physical geometry and selected ordered set; only map cached' if scope=='C' else 'none; map rebuilt each request',
                    map_cpu_bytes=max((r['cpu_bytes'] for r in maps),default=0),map_device_bytes=max((r['device_bytes'] for r in maps),default=0),
                    dominant_instrumented_stage=dominant,dominant_instrumented_stage_ms=medians[dominant],
                    checkpoint_sha256=a['operator_sha256'],normalizer_sha256=a['normalizer_sha256'],
                    selected_indices_sha256=a['selected_indices_sha256'],metric_target='field',
                    code_sha256=doc['source_hashes']['car_request.py'],reconstruction_code_sha256=doc['api_source_sha256'],
                    raw_source=str((directory/'timing_repeats.csv').relative_to(OUT)),condition=condition,
                    full_alias='one measured series; both old/new labels refer to it' if impl=='shared' else None,
                    reconstruction_scope='separately instrumented complete output pipeline including decode and transfers; not a stage-summed total',
                    preparation_note='geometry scorer/sampler preparation is inside each request; model loading is separately recorded',
                    execution_note='C old = rule-equivalent cached CPU sparse map; C new = cached GPU sparse map' if scope=='C' else 'native model; eager; FP32')
                results.append(result)
                if scope=='C' and variant!='full':
                    first_profile=[r for r in profiles if r['condition']==condition and not r['cache_hit']]
                    first_pipe=next(r['ms'] for r in first_profile if r['component']=='reconstruction_pipeline_ms')
                    map_info=next(r for r in maps if not r['cache_hit'])
                    for uses in protocol['reuse_counts']:
                        reuse.append(dict(backbone=arch,variant=variant,implementation=impl,case_id=doc['case_id'],
                            budget_fraction=doc['budget_fraction'],sampling_seed=doc['sampling_seed'],uses=uses,
                            first_request_ms=result['first_request_ms'],subsequent_request_median_ms=direct,
                            estimated_total_ms=result['first_request_ms']+(uses-1)*direct,
                            estimated_per_request_ms=(result['first_request_ms']+(uses-1)*direct)/uses,
                            first_reconstruction_pipeline_ms=first_pipe,subsequent_reconstruction_pipeline_median_ms=result['reconstruction_ms'],
                            cpu_map_build_ms=map_info['timings']['cpu_plan_total_ms'],
                            map_h2d_ms=map_info['timings'].get('map_h2d_ms',0.),map_cpu_bytes=result['map_cpu_bytes'],map_device_bytes=result['map_device_bytes'],
                            scope='C',cost_kind='derived first + (uses-1)*measured steady median; not a new total-wall measurement',
                            cache_key='dataset/rule/physical geometry identity/reference bytes+dtype/selected ordered indices/surface mask'))
        assert count==330,(scope,count)
    assert len(results)==990
    emit_csv(OUT/'results/reconstruction_summary.csv',results)
    emit_csv(OUT/'results/car_timing_repeats.csv',all_repeats)
    emit_csv(OUT/'results/car_stage_profiles.csv',all_profiles)
    emit_csv(OUT/'results/car_reuse_costs.csv',reuse)
    paired={(r['scope'],r['backbone'],r['variant'],r['case_id'],r['budget_fraction'],r['sampling_seed'],r['implementation']):r for r in results}
    speeds=[]
    for row in results:
        if row['implementation']!='old':continue
        other=paired[(row['scope'],row['backbone'],row['variant'],row['case_id'],row['budget_fraction'],row['sampling_seed'],'new')]
        assert row['checkpoint_sha256']==other['checkpoint_sha256'] and row['selected_indices_sha256']==other['selected_indices_sha256']
        speeds.append({k:row[k] for k in ['dataset','backbone','variant','case_id','budget_fraction','sampling_seed','scope']}|
            dict(old_direct_total_ms=row['direct_total_ms'],new_direct_total_ms=other['direct_total_ms'],
                 total_speedup=row['direct_total_ms']/other['direct_total_ms'],
                 old_reconstruction_ms=row['reconstruction_ms'],new_reconstruction_ms=other['reconstruction_ms'],
                 reconstruction_speedup=row['reconstruction_ms']/other['reconstruction_ms']))
    emit_csv(OUT/'results/car_old_new_speedups.csv',speeds)
    # Plotting rows pool formal observations across the fixed cases/seeds.
    group=defaultdict(list)
    for r in results:group[(r['scope'],r['backbone'],r['variant'],r['budget_fraction'],r['implementation'])].append(r)
    repeats=defaultdict(list)
    for r in all_repeats:
        if r['kind']=='formal':
            arch,variant,impl=r['condition'].split(':')
            repeats[(r['scope'],arch,variant,r['budget_fraction'],impl)].append(float(r['total_ms']))
    grouped=[]
    for k,values in sorted(group.items(),key=str):
        scope,arch,variant,budget,impl=k;times=repeats[k];expected=300 if variant=='full' else 900;assert len(times)==expected
        grouped.append(dict(scope=scope,backbone=arch,variant=variant,method=LABELS[variant],budget_fraction=budget,implementation=impl,
            direct_total_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),formal_observations=len(times),
            reconstruction_ms=float(np.median([r['reconstruction_ms'] for r in values])),
            peak_memory_allocated_bytes=max(r['peak_memory_allocated_bytes'] for r in values),
            peak_memory_reserved_bytes=max(r['peak_memory_reserved_bytes'] for r in values),
            old_new_field_max_abs=max(r['old_new_field_max_abs'] for r in values),case_count=3,sampling_n=0 if variant=='full' else 3,
            timing_aggregation='pooled100 formal repeats across3 fixed cases and3 seeds; Full3 cases only',
            first_request_ms=float(np.median([r['first_request_ms'] for r in values])),
            output_dtype='float64',output_device='GPU' if scope=='B' else 'CPU'))
    emit_csv(OUT/'results/car_latency_grouped.csv',grouped)
    return results,grouped,speeds


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    (OUT/'results').mkdir(exist_ok=True);protocol=read(OUT/'config/car_protocol.json');state=coverage(protocol)
    state.update(generated_at_unix=time.time(),protocol_sha256=sha(OUT/'config/car_protocol.json'))
    emit_json(OUT/'results/car_matrix_coverage.json',state)
    if args.partial:
        print(json.dumps(state,indent=2));return
    assert state['complete'],'Locked matrix incomplete; use --partial only for a coverage receipt'
    accuracy,budgets,rauec,correctness=aggregate_accuracy(protocol)
    from check_car_rauec import audit as independent_rauec_audit
    independent_rauec_audit()
    conditions,latency,speeds=aggregate_timing(protocol,accuracy)
    from plot_car import generate
    generate(OUT)
    from write_car_report import generate as write_report
    write_report(OUT)
    emit_json(OUT/'results/car_delivery_receipt.json',dict(execution_status='completed',evidence_status='verified',
        accuracy_pairs=7400,beta_pairs=1800,timing_conditions=990,formal_observations=99000,
        raw_accuracy_sha256=sha(OUT/'records/car_accuracy/pairs.jsonl'),
        raw_repeats_sha256=sha(OUT/'results/car_timing_repeats.csv'),
        figures='PDF/SVG/300dpiPNG generated from results/*.csv; render QA required separately'))
    print('Car aggregate completed:7400 old/new pairs,990 timing conditions and99000 formal observations')


if __name__=='__main__':main()
