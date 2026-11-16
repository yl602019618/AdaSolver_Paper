"""Select PI from calibration and report complete paired budget statistics."""
from pathlib import Path
from collections import defaultdict
import argparse,json,math,statistics

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
BUDGETS=[.375,.5,.75]

def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def area(v):return .0625*v[0]+.1875*v[1]+.125*v[2]

def collect(stage):
    records={}
    for part in range(4):
        folder=ROOT/f'{stage}_part{part}'
        manifest=read(folder/'run_manifest.json')
        assert manifest['status']=='complete' and manifest['min_true_cl']==.02
        for path in folder.glob('*_raw.jsonl'):
            for line in path.read_text().splitlines():
                row=json.loads(line);key=(row['method'],row['sample_id'],row['seed'],row['budget_fraction'])
                assert row['truth']['Cl']>.02 and row['truth_access'] is False
                if key in records:
                    assert row['selected_indices_hash']==records[key]['selected_indices_hash']
                    for metric in ['pressure_field_relative_l2','ratio_abs_error']:
                        assert row[metric]==records[key][metric]
                else:records[key]=row
    return list(records.values())

def summarize(rows,metric):
    groups=defaultdict(list)
    for row in rows:groups[row['method']].append(row)
    seeds=sorted({r['seed'] for r in groups['uniform']})
    expected={(r['sample_id'],r['seed'],r['budget_fraction']) for r in groups['uniform']}
    result={}
    for name,rr in groups.items():
        assert {(r['sample_id'],r['seed'],r['budget_fraction']) for r in rr}==expected
        invalid=sum(r[metric] is None or not math.isfinite(r[metric]) for r in rr)
        item={'config':rr[0]['resolved_method_config'],'cases':len({r['sample_id'] for r in rr}),
              'seeds':seeds,'records':len(rr),'invalid_metrics':invalid}
        if invalid:
            item['rauec_mean']=None;result[name]=item;continue
        curves={str(s):[statistics.mean(r[metric] for r in rr if r['seed']==s and r['budget_fraction']==b) for b in BUDGETS] for s in seeds}
        mean=[statistics.mean(curves[str(s)][i] for s in seeds) for i in range(3)]
        std=[statistics.stdev(curves[str(s)][i] for s in seeds) if len(seeds)>1 else 0. for i in range(3)]
        item.update(per_seed_mean_errors=curves,mean=mean,std=std)
        result[name]=item
    for name,item in result.items():
        if item['invalid_metrics']:continue
        scores=[area(item['per_seed_mean_errors'][str(s)])/area(result['uniform']['per_seed_mean_errors'][str(s)]) for s in seeds]
        item.update(rauec_mean=statistics.mean(scores),rauec_std=statistics.stdev(scores) if len(seeds)>1 else 0.,
                    per_seed_rauec=dict(zip(map(str,seeds),scores)),
                    pooled_rauec=area(item['mean'])/area(result['uniform']['mean']))
    return result

def main(stage):
    rows=collect(stage)
    summaries={metric:summarize(rows,key) for metric,key in [('cdcl','ratio_abs_error'),('field','pressure_field_relative_l2')]}
    dump(ROOT/(stage+'_summary.json'),summaries)
    ranking=sorted((n for n,v in summaries['cdcl'].items() if v['rauec_mean'] is not None),
                   key=lambda n:(summaries['cdcl'][n]['rauec_mean'],summaries['cdcl'][n]['rauec_std']))
    print(stage,[(n,round(summaries['cdcl'][n]['rauec_mean'],6)) for n in ranking[:12]],flush=True)
    if stage=='screen':
        shortlist=[]
        for family in ['pi','pi_voronoi']:
            names=[n for n in ranking if n.startswith('pi_candidate_') and summaries['cdcl'][n]['config']['method']==family][:3]
            shortlist.extend(summaries['cdcl'][n]['config'] for n in names)
        dump(ROOT/'configs/selection.json',read(ROOT/'configs/baselines.json')+shortlist)
        dump(ROOT/'screen_shortlist.json',{'names':[c['name'] for c in shortlist],'selection_metric':'mean per-seed Cd/Cl rAUEC'})
        ids=read(ROOT/'selection_ids.json')['sample_ids']
        for i in range(4):dump(ROOT/f'selection_ids_part{i}.json',{'sample_ids':ids[i::4]})
    elif stage=='selection':
        names=[n for n in ranking if n!='uniform']
        assert names
        chosen=summaries['cdcl'][names[0]]
        cfg=dict(chosen['config'],name='pi_cdcl_tuned',target='cdcl',status='frozen_after_selection')
        dump(ROOT/'configs/pi_cdcl_tuned.json',cfg)
        final=read(WORK/'airfoil_lift_threshold/configs.json')+[cfg]
        assert len({c['name'] for c in final})==len(final)==8
        dump(ROOT/'configs/frozen.json',final)
        dump(ROOT/'frozen_selection.json',{'source_candidate':names[0],'configuration':cfg,
             'selection_cdcl':chosen,'selection_field_diagnostic':summaries['field'][names[0]],
             'criterion':'lowest mean per-seed Cd/Cl rAUEC on the selection set',
             'confirmation_used_for_selection':False,'test_used_for_selection':False})
        for split in ['confirmation','test']:
            ids=read(ROOT/(split+'_ids.json'))['sample_ids']
            for i in range(4):dump(ROOT/f'{split}_ids_part{i}.json',{'sample_ids':ids[i::4]})
    else:
        with (ROOT/(stage+'_raw.jsonl')).open('w') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    return summaries

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen','selection','confirmation','test'])
    main(p.parse_args().stage)
