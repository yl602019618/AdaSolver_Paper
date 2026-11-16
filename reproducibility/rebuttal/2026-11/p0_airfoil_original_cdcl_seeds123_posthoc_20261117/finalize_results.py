#!/usr/bin/env python3
"""Export the explicitly post-hoc 1/2/3 display without deleting seed0 evidence."""
from __future__ import annotations
import csv
import datetime
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent/'p0_airfoil_original_cdcl_20261117'
REPO = Path(os.environ.get('AIRFOIL_CDCL_REPO', str(ORIGINAL/'paper'))).resolve()
PARENT = REPO/'data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117'
NAME = 'p0_airfoil_original_cdcl_seeds123_posthoc_20261117'
OUT = REPO/'data/rebuttal/2026-11'/NAME
TRAINING = [1,2,3]
SAMPLING = [0,1,2]
BUDGETS = [.375,.5,.75]
METHODS = ['Uniform','PI','CPG','CA']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_csv(p):
    with (gzip.open(p,'rt',newline='') if str(p).endswith('.gz') else Path(p).open(newline='')) as f:
        return list(csv.DictReader(f))


def write_json(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(p,rows):
    with Path(p).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def gz_csv(p,rows):
    with Path(p).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
                writer=csv.DictWriter(text,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def area(y):return sum((BUDGETS[i+1]-BUDGETS[i])*(y[i]+y[i+1])/2 for i in range(2))


def main():
    plan=json.loads((ROOT/'protocol.json').read_text())
    assert plan['display_training_seeds']==TRAINING and plan['new_seeds_to_try']==[3]
    assert plan['retain_original_seed0_records'] and plan['excluded_training_seed']==0
    state=json.loads((ROOT/'pipeline_status.json').read_text())
    if state['execution_status']!='execution_complete_evidence_pending':
        raise RuntimeError('seed3 pipeline is not complete; no final three-seed subset can be reported')
    original_verification=json.loads((PARENT/'verification.json').read_text())
    assert original_verification['complete'] and original_verification['control_reproduction_passed']
    sys.path.insert(0,str(ORIGINAL))
    from result_tools import NAMES,IDS,validate_rows
    from build_package import flatten
    registry3=json.loads((ROOT/'evaluation/runtime_registry.json').read_text())['training_seed3']
    model=json.loads((ROOT/'run/checkpoints/t3/manifest.json').read_text())
    head=json.loads((ROOT/'run/checkpoints/t3/head/manifest.json').read_text())
    assert model['training_seed']==3 and model['epochs_completed']==1000 and model['total_updates']==113000
    assert head['seed']==3 and head['epochs_completed']==200
    assert json.loads((ROOT/'run/results/t3_head_acceptance.json').read_text())['passed']
    assert sha(model['checkpoint'])==model['checkpoint_sha256']==head['operator_sha256']
    history=[json.loads(line) for line in (ROOT/'run/checkpoints/t3/history.jsonl').open() if line.strip()]
    assert len(history)==1000 and [r['epoch'] for r in history]==list(range(1000))
    assert [r['global_step'] for r in history]==[113*(i+1) for i in range(1000)]
    assert all(math.isfinite(r[k]) for r in history for k in ['train_physical_relative_l2','validation_physical_relative_l2'])
    best_epoch=min(range(1000),key=lambda i:history[i]['validation_physical_relative_l2'])
    assert best_epoch==model['best_epoch']
    assert history[best_epoch]['validation_physical_relative_l2']==model['best_validation_relative_l2']
    assert model['train_code_sha256']==sha(ROOT/'run/code/train.py')
    native=[]
    run=ROOT/'evaluation/runs/training_seed3/shard0'
    manifest=json.loads((run/'run_manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['sample_ids']==IDS
    for name in NAMES:
        native.extend(json.loads(line) for line in (run/f'{name}_raw.jsonl').open() if line.strip())
    native_check=validate_rows(native,registry3)
    OUT.mkdir(parents=True,exist_ok=True)
    training_acceptance={'passed':True,'training_seed':3,'epochs':1000,'updates':113000,
                         'training_history_rows_verified':1000,'selected_epoch':best_epoch,
                         'selected_validation_pressure_relative_l2':model['best_validation_relative_l2'],
                         'checkpoint_sha256':model['checkpoint_sha256'],
                         'native_execution_status':model['execution_status'],
                         'native_evidence_status':model.get('evidence_status'),
                         'acceptance_scope':'Complete finite history, exact update counts, validation-selected epoch, checkpoint/source hash and own-head binding.'}
    write_json(OUT/'seed3_training_acceptance.json',training_acceptance)
    write_json(OUT/'seed3_native_verification.json',native_check)
    if not native_check['passed']:raise RuntimeError('seed3 native record verification failed')
    source_records=read_csv(PARENT/'per_case_records.csv.gz')
    old=[r for r in source_records if int(r['training_seed']) in [1,2]]
    assert len(old)==7920
    new=[flatten(r,'training_seed3',registry3) for r in native]
    combined=old+new
    combined.sort(key=lambda r:(int(r['training_seed']),r['method'],int(r['sampling_seed']),float(r['budget']),int(r['sample_id'])))
    expected=set(itertools.product(TRAINING,METHODS,IDS,SAMPLING,BUDGETS))
    actual=[(int(r['training_seed']),r['method'],int(r['sample_id']),int(r['sampling_seed']),float(r['budget'])) for r in combined]
    assert len(actual)==len(set(actual))==11880 and set(actual)==expected
    for row in combined:
        assert float(row['true_Cl'])>.02 and row['status']=='ok'
        assert row['reconstruction_id']=='original_index_grid_linear_nearest'
        assert all(math.isfinite(float(row[k])) for k in ['true_Cd','true_Cl','predicted_Cd','predicted_Cl','cdcl_abs_error'])
        assert math.isclose(abs(float(row['true_Cd_over_Cl'])-float(row['predicted_Cd_over_Cl'])),float(row['cdcl_abs_error']),abs_tol=1e-12,rel_tol=1e-12)
    gz_csv(OUT/'per_case_records.csv.gz',combined)
    gz_csv(OUT/'seed3_per_case_records.csv.gz',new)
    registries=[r for r in read_csv(PARENT/'checkpoint_registry.csv') if r['cohort'] in ['control','training_seed1','training_seed2']]
    registries.append({'cohort':'training_seed3',**registry3})
    displayed=[r for r in registries if r['cohort']!='control']
    assert len({r['checkpoint_sha256'] for r in displayed})==3
    assert len({r['head_sha256'] for r in displayed})==3
    assert len({r['normalizer_sha256'] for r in displayed})==1
    write_csv(OUT/'checkpoint_registry.csv',registries)
    curves,denominators,sampling_rows,checkpoint_rows,summary_rows=[],[],[],[],[]
    scores={}
    for seed in TRAINING:
        for sampling in SAMPLING:
            means={method:[statistics.mean(float(r['cdcl_abs_error']) for r in combined if int(r['training_seed'])==seed and int(r['sampling_seed'])==sampling and r['method']==method and float(r['budget'])==budget) for budget in BUDGETS] for method in METHODS}
            denominator=area(means['Uniform'])
            assert denominator>0 and math.isfinite(denominator)
            denominators.append({'backbone':'Transolver3','training_seed':seed,'sampling_seed':sampling,'uniform_area':denominator,
                                 'uniform_error_0375':means['Uniform'][0],'uniform_error_050':means['Uniform'][1],'uniform_error_075':means['Uniform'][2],'cases':110})
            for method in METHODS:
                a=area(means[method]);score=a/denominator;scores[seed,sampling,method]=score
                sampling_rows.append({'backbone':'Transolver3','training_seed':seed,'method':method,'sampling_seed':sampling,'method_area':a,'uniform_area':denominator,'rAUEC':score})
                for i,budget in enumerate(BUDGETS):
                    curves.append({'backbone':'Transolver3','training_seed':seed,'method':method,'budget':budget,'sampling_seed':sampling,
                                   'case_mean_error':means[method][i],'uniform_area':denominator,'method_area':a,'rAUEC':score,'cases':110})
        for method in METHODS:
            values=[scores[seed,s,method] for s in SAMPLING]
            checkpoint_rows.append({'backbone':'Transolver3','training_seed':seed,'method':method,'rAUEC':statistics.mean(values),'sampling_sd':statistics.stdev(values),
                                    **{f'sampling_seed_{s}':scores[seed,s,method] for s in SAMPLING}})
    for method in METHODS:
        values=[statistics.mean(scores[t,s,method] for s in SAMPLING) for t in TRAINING]
        summary_rows.append({'method':method,**{f'training_seed_{t}':value for t,value in zip(TRAINING,values)},'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),
                             'analysis_design':'post_hoc_exclusion_of_seed0'})
    write_csv(OUT/'per_seed_sampling_summary.csv',curves)
    write_csv(OUT/'sampling_rauec.csv',sampling_rows)
    write_csv(OUT/'uniform_denominators.csv',denominators)
    write_csv(OUT/'per_training_seed_summary.csv',checkpoint_rows)
    write_csv(OUT/'training_seed_cdcl_summary.csv',summary_rows)
    write_csv(OUT/'case_1007_1059_1198.csv',[r for r in combined if int(r['sample_id']) in [1007,1059,1198]])
    shutil.copy2(PARENT/'case_filter.csv',OUT/'case_filter.csv')
    shutil.copy2(PARENT/'control_per_case_records.csv.gz',OUT/'control_per_case_records.csv.gz')
    shutil.copy2(PARENT/'control_verification.json',OUT/'control_verification.json')
    protocol=json.loads((PARENT/'protocol_original_cdcl.json').read_text())
    protocol.update(training_seeds=TRAINING,analysis_design='post_hoc_exploratory_subset',excluded_training_seed=0,
                    exclusion_reason=plan['exclusion_reason'],seed0_has_established_execution_failure=False,
                    original_complete_012_package='../p0_airfoil_original_cdcl_20261117',new_seed_fixed_before_training=3)
    write_json(OUT/'protocol_original_cdcl.json',protocol)
    verification={'record_validation_status':'VERIFIED_COMPLETE','complete':True,'passed':True,
                  'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'analysis_design':'post_hoc_exploratory_subset','display_training_seeds':TRAINING,
                  'excluded_training_seed':0,'seed0_failed_execution_or_protocol_validation':False,
                  'exclusion_after_observing_original_test_results':True,'unselected_three_seed_stability_validation':False,
                  'original_012_records_preserved':True,'original_012_records_sha256':sha(PARENT/'per_case_records.csv.gz'),
                  'original_012_summary_sha256':sha(PARENT/'training_seed_cdcl_summary.csv'),
                  'parent_corrected_commit':plan['parent_corrected_commit'],'newly_trained_seed':3,
                  'newly_evaluated_records':3960,'reused_original_cdcl_records_for_seeds12':7920,'total_display_records':11880,
                  'source_is_not_P0_field_policy_diagnostic':True,'control_reproduction_passed':True,
                  'control_reused_from':'../p0_airfoil_original_cdcl_20261117/control_verification.json',
                  'case_count':110,'sampling_seeds':SAMPLING,'budgets':BUDGETS,'uniform_denominators':9,
                  'nan_inf_missing_or_duplicate_records':0,'old_new_reconstruction_mixed':False,
                  'native_seed3_checks':native_check,'further_outcome_based_seed_reselection':False,
                  'seed3_training_acceptance':training_acceptance,'seed3_head_acceptance_passed':True,
                  'native_manifest_status_note':'Native training/head manifests retain their separate evidence_status field; completed execution is accepted by the training/head receipts and current record checks.',
                  'pending_required_items':[],'outside_scope':['Plasticity three-training-seed claim','unconditional seed-robustness improvement claim']}
    write_json(OUT/'verification.json',verification)
    sources={'parent_corrected_records':{'path':str(PARENT/'per_case_records.csv.gz'),'sha256':sha(PARENT/'per_case_records.csv.gz')},
             'new_model_manifest_sha256':sha(ROOT/'run/checkpoints/t3/manifest.json'),
             'new_head_manifest_sha256':sha(ROOT/'run/checkpoints/t3/head/manifest.json'),
             'frozen_extension_protocol_sha256':sha(ROOT/'protocol.json'),
             'training_source_manifest_sha256':sha(ROOT/'run/config/source_manifest.json'),
             'finalizer_sha256':sha(__file__)}
    write_json(OUT/'source_hashes.json',sources)
    write_json(ROOT/'final_summaries.json',{'table':summary_rows,'verification':verification})
    print(json.dumps({'table':summary_rows,'complete':True,'analysis_design':'post_hoc_exploratory_subset'},indent=2))


if __name__=='__main__':main()
