"""Export verified LinearNO Car PI/CA accuracy, trained head and direct timing."""
import csv
import gzip
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import itertools
import datetime
from common import ROOT,E3,sha256,write_json
from accuracy_tools import load_runs,validate_runs,aggregate,expected_configs

NAME='linearno_car_pi_ca_20261117'
PAPER=ROOT/'paper'
DATA=PAPER/'data/rebuttal/2026-11'/NAME
REPRO=PAPER/'reproducibility/rebuttal/2026-11'/NAME


def csv_read(path):return list(csv.DictReader(Path(path).open()))


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def csv_gzip(path,rows):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    import numpy as np
    runs=load_runs();native=validate_runs(runs);assert native['passed']
    control=json.loads((ROOT/'accuracy_verification.json').read_text());assert control['passed']
    head_validation=json.loads((ROOT/'head_acceptance.json').read_text());assert head_validation['passed']
    DATA.mkdir(parents=True,exist_ok=True);REPRO.mkdir(parents=True,exist_ok=True)
    summaries,curves,denominators=aggregate(runs)
    csv_write(DATA/'accuracy_summary.csv',summaries);csv_write(DATA/'per_sampling_seed_budget.csv',curves);csv_write(DATA/'uniform_denominators.csv',denominators)
    registry=[];records=[];primary=[];sources={}
    for variant,run in runs.items():
        m=run['manifest'];config=m['config']
        registry.append({'variant':variant,'method':config['method'],'policy_target':config['selection_target'],
                         'beta':config.get('beta',''),'records':len(run['rows']),'run_id':m['run_id'],
                         'checkpoint_sha256':m['operator_sha256'],'normalizer_sha256':m['normalizer_sha256'],
                         'head_sha256':m['head_sha256'],'split_hash':m['split_hash'],'source_record_path':str(run['path']/'raw.jsonl'),
                         'source_record_sha256':sha256(run['path']/'raw.jsonl')})
        sources[str(run['path']/'run_manifest.json')]=sha256(run['path']/'run_manifest.json')
        for r in run['rows']:
            row={'backbone':'LinearNO','training_seed':0,'variant':variant,'method':r['method'],'policy_target':r['selection_target'],
                 'beta':config.get('beta',''),'sample_id':r['sample_id'],'sampling_seed':r['seed'],'budget':r['budget_fraction'],
                 **{k:r[k] for k in ['field_error','velocity_rel_l2','pressure_rel_l2','cd_relative_error','cd_pred','cd_gt',
                                    'cd_pressure_pred','cd_shear_pred','cd_pressure_gt','cd_shear_gt','full_point_count','selected_count',
                                    'coarse_count','surface_count','surface_selected_count','selected_indices_sha256','coarse_indices_sha256',
                                    'operator_sha256','normalizer_sha256','split_hash','mesh_sha256','status','cd_status']},
                 'head_sha256':m['head_sha256'],'reconstruction_id':'original_physical_knn4_idw_surface_pressure'}
            records.append(row)
            targets=['field','cd'] if variant in ['full','uniform'] else [config['selection_target']]
            for target in targets:
                metric='field_error' if target=='field' else 'cd_relative_error'
                primary.append({'backbone':'LinearNO','training_seed':0,'target':target,'variant':variant,'method':r['method'],
                                'beta':config.get('beta',''),'sample_id':r['sample_id'],'sampling_seed':'' if variant=='full' else r['seed'],
                                'budget':r['budget_fraction'],'metric':metric,'error':r[metric],
                                'role':'metric_specific_policy' if variant not in ['full','uniform'] else 'shared_reference'})
    assert len(records)==8200 and len(primary)==9200
    csv_write(DATA/'run_registry.csv',registry);csv_gzip(DATA/'per_case_records.csv.gz',records);csv_gzip(DATA/'primary_metric_records.csv.gz',primary)
    full=[r for r in records if r['variant']=='full'];csv_write(DATA/'full_reference.csv',full)
    raw_timing=[];condition_summaries=[];alignments=[];hardware=[]
    timing_protocol=json.loads((ROOT/'inference_timing_protocol.json').read_text());timing_cases=timing_protocol['timing_cases']
    timing_checks={}
    for target in ['field','cd']:
        folder=ROOT/'records/formal_inference'/target;manifest=json.loads((folder/'run_manifest.json').read_text())
        if manifest['status']!='completed':raise RuntimeError(f'formal inference timing incomplete: {target}')
        rows=csv_read(folder/'timing_repeats.csv');reported=csv_read(folder/'latency_summary.csv')
        assert len(rows)==13320 and len(reported)==111
        expected={('full',case,None,1.) for case in timing_cases}
        expected|={(name,case,seed,budget) for name in ['uniform','pi_'+target,'cpg_'+target,'ca_'+target]
                    for case in timing_cases for seed in [0,1,2] for budget in [.375,.5,.75]}
        groups={};observed=set();bad=[]
        for r in rows:
            key=(r['variant'],r['case_id'],None if r['sampling_seed']=='' else int(r['sampling_seed']),float(r['budget']))
            item=groups.setdefault(key,{'warmup':{},'formal':{}});phase=r['phase'];rep=int(r['repeat']);value=float(r['wall_ms'])
            assert math.isfinite(value) and value>0 and rep not in item[phase]
            item[phase][rep]=value;observed.add(key)
            raw_timing.append({'backbone':'LinearNO','training_seed':0,**r})
        assert observed==expected
        reported_map={(r['variant'],r['case_id'],None if r['sampling_seed']=='' else int(r['sampling_seed']),float(r['budget'])):r for r in reported}
        max_difference=0.
        for key,g in groups.items():
            assert set(g['warmup'])==set(range(20)) and set(g['formal'])==set(range(100))
            values=[g['formal'][i] for i in range(100)];rr=reported_map[key]
            wanted={'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),
                    'mean_ms':statistics.mean(values),'repeat_sd_ms':statistics.stdev(values)}
            for k,v in wanted.items():max_difference=max(max_difference,abs(v-float(rr[k])))
            assert max_difference<1e-10
            condition_summaries.append({'backbone':'LinearNO','training_seed':0,**rr})
        alignment=json.loads((folder/'alignment.json').read_text());assert alignment['passed'] and len(alignment['conditions'])==111
        assert all(max(c['native_metric_absolute_differences'].values())<=2e-6 for c in alignment['conditions'])
        alignments.extend(alignment['conditions'])
        hardware.append({'target':target,'physical_gpu':manifest['physical_gpu'],'cpu_affinity':manifest['cpu_affinity'],'environment':manifest['environment']})
        timing_checks[target]={'passed':True,'conditions':111,'formal_repetitions':11100,'warmups':2220,
                               'summary_max_recompute_difference':max_difference,'metric_alignment_conditions':111,
                               'maximum_native_metric_difference':max(max(c['native_metric_absolute_differences'].values()) for c in alignment['conditions'])}
        shutil.copy2(folder/'run_manifest.json',DATA/f'inference_{target}_manifest.json')
        shutil.copy2(folder/'gpu_telemetry.json',DATA/f'inference_{target}_gpu_telemetry.json')
    csv_gzip(DATA/'inference_repeats.csv.gz',raw_timing);csv_write(DATA/'inference_conditions.csv',condition_summaries)
    write_json(DATA/'inference_alignment.json',{'passed':True,'conditions':alignments});write_json(DATA/'hardware.json',hardware)
    grouped=[]
    for target in ['field','cd']:
        selected=[r for r in condition_summaries if r['target']==target]
        full_mean=statistics.mean(float(r['median_ms']) for r in selected if r['method']=='fullmesh')
        for variant,budget in sorted({(r['variant'],float(r['budget'])) for r in selected}):
            group=[r for r in selected if r['variant']==variant and float(r['budget'])==budget]
            values=[float(r['median_ms']) for r in group]
            grouped.append({'target':target,'variant':variant,'method':group[0]['method'],'budget':budget,'physical_gpu':group[0]['physical_gpu'],
                            'conditions':len(group),'mean_condition_median_ms':statistics.mean(values),
                            'condition_median_sd_ms':statistics.stdev(values),'same_gpu_full_mean_median_ms':full_mean,
                            'full_over_method_latency_ratio':full_mean/statistics.mean(values),
                            'aggregation':'mean of3Full or9case/sampling condition medians; no crossGPU pooling'})
    csv_write(DATA/'inference_budget_summary.csv',grouped)
    operator_path=E3/'checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json';operator=json.loads(operator_path.read_text())
    hp=ROOT/'checkpoints/seed0/head/ca_pressure_linearno_trainingseed0_headseed0_manifest.json';head=json.loads(hp.read_text())
    shutil.copy2(head['checkpoint'],DATA/'linearno_car_pressure_head_seed0.pt')
    portable={**head,'checkpoint':'linearno_car_pressure_head_seed0.pt','source_checkpoint_path':head['checkpoint']}
    write_json(DATA/'head_inference_manifest.json',portable);write_json(DATA/'head_training_manifest.json',head)
    csv_write(DATA/'head_training_history.csv',head['training_history'])
    csv_write(DATA/'checkpoint_registry.csv',[{'backbone':'LinearNO','training_seed':0,'head_seed':0,
       'checkpoint_path':operator['checkpoint'],'checkpoint_sha256':operator['checkpoint_sha256'],'operator_manifest_path':str(operator_path),
       'normalizer_path':operator['normalizer'],'normalizer_sha256':operator['normalizer_sha256'],'backbone_epochs':400,'backbone_updates':142000,
       'backbone_parameters':3852420,'head_checkpoint':'linearno_car_pressure_head_seed0.pt','head_sha256':head['head_sha256'],
       'head_epochs':200,'head_parameters':16513,'head_validation_mse':head['best_validation_mse'],
       'head_feature_source':'own official native block0 on10% inputs','reconstruction':'original k4physical IDW'}])
    split=json.loads((ROOT/'splits/seed0.json').read_text());write_json(DATA/'splits.json',split)
    csv_write(DATA/'test_cases.csv',[{'case_id':sid,'fold':0,'included':True} for sid in split['test']])
    write_json(DATA/'frozen_policies.json',expected_configs())
    scope=json.loads((ROOT/'training_tasks.json').read_text())['scope']
    write_json(DATA/'protocol.json',{'scope':scope,'training_seeds':[0],'head_seed':0,'sampling_seeds':[0,1,2],
         'budgets':[.375,.5,.75],'test_case_count':100,'head_training_cases':710,'head_validation_cases':39,'head_epochs':200,
         'statistics':'case means ->trapezoidal budget area ->same checkpoint/sampling Uniform normalization ->sampling mean andsample SD',
         'training_sd':'not estimated from one training run','inference_timing':timing_protocol,
         'policy_identity':'existing E5 frozen PI/CA Field/Cd and E3 CPGbeta0/1 controls; no test retuning',
         'full_reference':'Full100% canonical execution, outside rAUEC'})
    verification={'status':'VERIFIED_COMPLETE','complete':True,'backbone_training_runs':1,'new_backbone_training_runs':0,'new_complete_heads':1,
        'head':head_validation,'accuracy':control,'new_pi_ca_records':3600,'control_records':4600,'native_records':8200,'primary_metric_views':9200,
        'inference_timing':timing_checks,'total_formal_conditions':222,'total_formal_repetitions':22200,'total_warmups':4440,
        'sampling_sd_is_not_training_sd':True,'metric_specific_policies_preserved':True,'field_policy_cd_diagnostics_not_used_as_primary_cd':True,
        'source_checkpoint_and_normalizer_unchanged':True,'pending_required_items':[],
        'outside_this_scope':['LinearNO Car independent training seeds1/2','B/C reconstruction execution scenarios','isolated per-method GPU memory'],
        'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(DATA/'verification.json',verification);shutil.copy2(ROOT/'head_acceptance.json',DATA/'head_acceptance.json')
    write_json(DATA/'source_hashes.json',{'run_manifests':sources,'bridge_code':{p.name:sha256(p) for p in ROOT.glob('*.py')},
          'frozen_inputs':json.loads((ROOT/'training_tasks.json').read_text())['source_files'],
          'official_repository':'https://github.com/HiPRL/LinearNO','official_commit':'3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269',
          'official_implementation_bundled':False})
    for name in ['common.py','linearno_ca.py','prepare_head_dataset.py','train_head.py','bind_splits.py','evaluate.py','run_stage.py','run_backbone.py',
                 'verify_interface.py','verify_timing.py','formal_timing.py','workflow.py','accuracy_tools.py','verify_accuracy.py','validate_trained_head.py',
                 'inference_request.py','benchmark_inference.py','package_results.py',
                 'training_tasks.json','seed0_missing_methods.json','evaluation_methods.json','pi_only.json','ca_only.json','controls.json',
                 'timing_protocol.json','inference_timing_protocol.json','interface_validation.json','timing_preflight.json']:
        shutil.copy2(ROOT/name,REPRO/name)
    for name in ['linearno_adapter.py','evaluate_linearno.py']:
        shutil.copy2(E3/'code'/name,REPRO/name)
    (DATA/'.gitattributes').write_text('* -text -filter\n');(REPRO/'.gitattributes').write_text('* -text -filter\n')
    print(json.dumps({'status':'VERIFIED_COMPLETE','accuracy_summary':summaries,'timing':timing_checks,'data_package':str(DATA)},indent=2))


if __name__=='__main__':main()
