"""Check all eight reference points against the 1,100 published case records."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parents[1]


def main():
    manifest=json.loads((ROOT/'training_manifest.json').read_text())
    assert manifest['epochs']==1000 and manifest['target_channel']==3
    assert manifest['nmin_count']==manifest['nfull']==11271 and manifest['nmin_fraction']==1.
    records=list(csv.DictReader((ROOT/'records.csv').open()))
    curves=list(csv.DictReader((ROOT/'budget_curves.csv').open()))
    plotted=list(csv.DictReader((PAPER/'figures/appendix/fullmesh_training_curve_points.csv').open()))
    denominators=json.loads((PAPER/'data/airfoil_pressure/protocol.json').read_text())['fullmesh_denominators']
    ids=set(json.loads((PAPER/'data/airfoil_pressure/protocol.json').read_text())['sample_ids'])
    keys={(int(r['sample_id']),int(r['seed']),float(r['budget_fraction'])) for r in records}
    assert len(records)==len(keys)==1100
    assert all(float(r['true_cl'])>.02 for r in records)
    for r in records:
        error=abs(float(r['predicted_cd'])/float(r['predicted_cl'])-float(r['true_cd'])/float(r['true_cl']))
        assert math.isclose(error,float(r['ratio_abs_error']),rel_tol=1e-10,abs_tol=1e-12)
    for curve in curves:
        metric=curve['metric'];field='pressure_field_relative_l2' if metric=='field' else 'ratio_abs_error'
        budget=float(curve['budget_percent'])/100
        seeds=[0] if budget==1 else [0,1,2]
        values=[]
        for seed in seeds:
            rows=[r for r in records if float(r['budget_fraction'])==budget and int(r['seed'])==seed]
            assert len(rows)==110 and {int(r['sample_id']) for r in rows}==ids
            values.append(statistics.mean(float(r[field]) for r in rows))
        mean=statistics.mean(values);sd=statistics.stdev(values) if len(values)>1 else 0.
        for key,number in [('raw_mean_error',mean),('raw_std_error',sd),
                           ('normalized_mean',mean/denominators[metric]),('normalized_std',sd/denominators[metric])]:
            assert math.isclose(float(curve[key]),number,rel_tol=1e-11,abs_tol=1e-13)
        line=next(r for r in plotted if r['benchmark']=='airfoil2d' and r['metric']==metric and float(r['budget_percent'])==budget*100)
        assert line==curve
    print('PASS: full-mesh pressure training metadata, 1,100 records, raw force ratios and eight plotted points.')


if __name__=='__main__':
    main()
