"""Independent rAUEC recomputation from per-case CSV (no aggregator import)."""
from __future__ import annotations
import argparse,csv,json,math,statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
B=(.375,.5,.75)


def area(curve):return sum((x1-x0)*(y0+y1)/2 for x0,x1,y0,y1 in zip(B,B[1:],curve,curve[1:]))

def self_test():
    uniform=[[1,2,3],[2,2,2],[1,1,1]];method=[[1,1,1],[1,2,3],[2,2,2]]
    expected=[float(Fraction(6,13)),float(Fraction(13,12)),2.]
    ratios=[area(x)/area(y) for x,y in zip(method,uniform)]
    assert all(math.isclose(x,y,rel_tol=1e-15) for x,y in zip(ratios,expected))
    assert math.isclose(statistics.mean(ratios),float(Fraction(553,468)),rel_tol=1e-15)
    pooled=area([statistics.mean(x) for x in zip(*method)])/area([statistics.mean(x) for x in zip(*uniform)])
    assert pooled==1. and not math.isclose(statistics.mean(ratios),pooled)
    return {'passed':True,'fixture':'analytical curves, not experiment results','per_seed_ratios':ratios,'expected_mean':float(Fraction(553,468)),
            'ratio_after_seed_average':pooled,'distinguishes_required_statistical_order':True}


def audit():
    groups=defaultdict(list)
    for r in csv.DictReader((ROOT/'results/car_per_case_metrics.csv').open(newline='')):
        if r['variant']=='full':continue
        k=tuple(r[x] for x in ['backbone','implementation','variant','metric','sampling_seed'])+(float(r['budget_fraction']),)
        groups[k].append(float(r['error']))
    curves={}
    for k,values in groups.items():assert len(values)==100;curves[k]=math.fsum(values)/100
    count=0;max_delta=0.;ratios=defaultdict(list)
    for r in csv.DictReader((ROOT/'results/car_rauec_seed.csv').open(newline='')):
        key=tuple(r[x] for x in ['backbone','implementation','variant','metric','sampling_seed'])
        base=(key[0],key[1],'uniform',key[3],key[4])
        value=area([curves[key+(b,)] for b in B])/area([curves[base+(b,)] for b in B])
        delta=abs(value-float(r['rAUEC']));assert delta<1e-12
        max_delta=max(max_delta,delta);count+=1;ratios[key[:4]].append(value)
    for r in csv.DictReader((ROOT/'results/car_rauec.csv').open(newline='')):
        k=tuple(r[x] for x in ['backbone','implementation','variant','metric']);values=ratios[k];assert len(values)==3
        assert abs(statistics.mean(values)-float(r['rAUEC_mean']))<1e-12
        assert abs(statistics.stdev(values)-float(r['rAUEC_sampling_sd']))<1e-12
    assert count==192,count
    result={'passed':True,'source':'car_per_case_metrics.csv','per_seed_ratios_recomputed':count,'maximum_difference':max_delta,
            'full_excluded':True,'sample_sd_ddof':1,'population_count':100,'self_test':self_test()}
    (ROOT/'results/car_rauec_independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    print(json.dumps(self_test() if a.self_test else audit(),indent=2))
