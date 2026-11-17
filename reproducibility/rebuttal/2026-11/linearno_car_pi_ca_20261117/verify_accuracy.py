"""Verify full coverage and reproduce the existing official LinearNO controls."""
import csv
import json
from pathlib import Path
from common import ROOT,E3,write_json
from accuracy_tools import load_runs,validate_runs,aggregate,label


def main():
    runs=load_runs();verification=validate_runs(runs)
    controls={};sources=[]
    operator=runs['full']['manifest']['operator_sha256']
    for folder in ['adaptive','full']:
        for path in (E3/'records'/folder).glob('*/run_manifest.json'):
            meta=json.loads(path.read_text())
            if meta.get('operator_sha256')!=operator or meta.get('status')!='complete':continue
            name=label(meta['config'])
            if name not in ['full','uniform','cpg_field','cpg_field_beta0','cpg_cd','cpg_cd_beta0']:continue
            assert meta['config']==runs[name]['manifest']['config']
            sources.append(str(path))
            for line in (path.parent/'raw.jsonl').open():
                r=json.loads(line);controls[name,r['sample_id'],r['seed'],r['budget_fraction']]=r
    maximum=0.;mismatches=[];checked=0
    for name in ['full','uniform','cpg_field','cpg_field_beta0','cpg_cd','cpg_cd_beta0']:
        for row in runs[name]['rows']:
            key=(name,row['sample_id'],row['seed'],row['budget_fraction']);old=controls[key]
            deltas={k:abs(row[k]-old[k]) for k in ['field_error','velocity_rel_l2','pressure_rel_l2','cd_relative_error']}
            maximum=max(maximum,max(deltas.values()));checked+=1
            if max(deltas.values())>2e-6 or row['selected_indices_sha256']!=old['selected_indices_sha256'] or row['coarse_indices_sha256']!=old['coarse_indices_sha256']:
                mismatches.append({'key':key,'metric_differences':deltas})
    assert checked==4600
    verification['original_control_reproduction']={'records':checked,'sources':sources,'maximum_absolute_metric_difference':maximum,'mismatches':mismatches,'passed':not mismatches}
    verification['passed']=verification['passed'] and not mismatches
    write_json(ROOT/'accuracy_verification.json',verification)
    if not verification['passed']:raise RuntimeError('Accuracy or control reproduction failed')
    summaries,curves,denominators=aggregate(runs)
    for name,rows in [('accuracy_summary.csv',summaries),('per_sampling_seed_budget.csv',curves),('uniform_denominators.csv',denominators)]:
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps({'passed':True,'native_records':verification['total_native_records'],'controls':verification['original_control_reproduction'],'summary':summaries},indent=2))


if __name__=='__main__':main()
