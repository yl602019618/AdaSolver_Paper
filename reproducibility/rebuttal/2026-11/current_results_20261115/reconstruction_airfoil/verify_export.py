#!/usr/bin/env python3
"""Independently verify exported current-results CSVs without model/data access."""
import argparse,collections,csv,json,gzip
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path);args=p.parse_args();base=args.data_dir;output=args.output_dir or base;output.mkdir(parents=True,exist_ok=True)
    def read(name):
        path=base/name
        opener=(lambda:gzip.open(str(path)+'.gz','rt')) if Path(str(path)+'.gz').exists() else path.open
        with opener() as f:return list(csv.DictReader(f))
    def equal(a,b):assert np.isclose(float(a),float(b),atol=1e-10,rtol=1e-12),(a,b)
    report={}
    for dataset in ['car','airfoil']:
        raw=read(dataset+'_raw_timing_repeats.csv');conditions=read(dataset+'_reconstruction_conditions.csv')
        by_condition=collections.defaultdict(list)
        for r in raw:
            group=r['source_group'] if dataset=='car' else r['group_id']
            by_condition[(group,r['condition'])].append(r)
        groups=collections.defaultdict(list)
        for r in conditions:
            group=str(Path(r['raw_source']).parent) if dataset=='car' else r['group_id']
            series=by_condition[(group,r['condition'])]
            formal=sorted([x for x in series if x['kind']=='formal'],key=lambda x:int(x['repeat_id']))
            warm=[x for x in series if x['kind']=='warmup']
            assert len(formal)==100 and len(warm)==20
            assert {int(x['repeat_id']) for x in formal}==set(range(100))
            values=[float(x['total_ms']) for x in formal]
            equal(np.median(values),r['direct_total_ms']);equal(np.percentile(values,95),r['p95_ms'])
            method=r['variant'] if dataset=='car' else r['method']
            key=(r['scope'],r['backbone'],method,float(r['budget_fraction']),r['implementation'])
            groups[key].extend(values)
            if method=='full':assert r['sampling_seed'] in ['', 'N/A']
        grouped=read(dataset+'_latency_grouped.csv')
        for r in grouped:
            method=r['variant'] if dataset=='car' else r['method']
            values=groups[(r['scope'],r['backbone'],method,float(r['budget_fraction']),r['implementation'])]
            equal(np.median(values),r['direct_total_ms']);equal(np.percentile(values,95),r['p95_ms'])
        seed_rows=read(dataset+'_sampling_seed_errors.csv')
        lookup={(r['backbone'],r['checkpoint_sha256'],r['implementation'],r['reconstruction_id'],r['method'],r['beta'],r['metric'],float(r['budget_fraction']),r['sampling_seed']):float(r['error_mean']) for r in seed_rows}
        for r in read(dataset+'_rauec.csv'):
            assert r['method']!='full';ratios=[]
            for s in ['0','1','2']:
                k=(r['backbone'],r['checkpoint_sha256'],r['implementation'],r['reconstruction_id'])
                y=[lookup[(*k,r['method'],r['beta'],r['metric'],b,s)] for b in [.375,.5,.75]]
                u=[lookup[(*k,'uniform','',r['metric'],b,s)] for b in [.375,.5,.75]]
                area=float(np.trapezoid(u,[.375,.5,.75]));ratio=float(np.trapezoid(y,[.375,.5,.75]))/area
                equal(area,r['matched_uniform_area_seed'+s]);equal(ratio,r['sampling_seed'+s]);ratios.append(ratio)
            equal(np.mean(ratios),r['mean']);equal(np.std(ratios,ddof=1),r['sampling_sd'])
        cache=read('car_reuse_costs.csv' if dataset=='car' else 'airfoil_cache_reuse_costs.csv')
        for r in cache:
            if dataset=='car':expected=float(r['first_request_ms'])+(int(r['uses'])-1)*float(r['subsequent_request_median_ms']);equal(expected,r['estimated_total_ms'])
            else:expected=float(r['first_request_ms'])+(int(r['requests'])-1)*float(r['steady_median_ms']);equal(expected,r['modeled_total_ms'])
        report[dataset]={'conditions':len(conditions),'pooled_summary_rows':len(grouped),'formal_repeat_rows':sum(r['kind']=='formal' for r in raw),'cache_rows':len(cache),'passed':True}
    assert report['car']['conditions']==990 and report['airfoil']['conditions']==1323
    (output/'export_validation.json').write_text(json.dumps({'status':'passed','checks':report,'scope':'CSV-only independent direct-repeat, pooled median/p95, matched rAUEC and cache estimate recomputation'},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
