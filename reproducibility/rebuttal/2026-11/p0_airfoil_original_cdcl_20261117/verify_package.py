#!/usr/bin/env python3
"""Recompute all public Cd/Cl statistics and coverage without a GPU or checkpoint."""
from __future__ import annotations
import argparse
import csv
import gzip
import itertools
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with (gzip.open(path, 'rt', newline='') if str(path).endswith('.gz') else Path(path).open(newline='')) as f:
        return list(csv.DictReader(f))


def area(values):
    budgets = [.375, .5, .75]
    return sum((budgets[i+1]-budgets[i])*(values[i]+values[i+1])/2 for i in range(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    protocol = json.loads((data/'protocol_original_cdcl.json').read_text())
    ids = protocol['sample_ids']
    assert len(ids) == len(set(ids)) == 110 and not set(ids)&{1041,1139}
    assert protocol['budgets'] == [.375,.5,.75]
    assert protocol['sampling_seeds'] == [0,1,2] and protocol['training_seeds'] == [0,1,2]
    root = data.parents[3]
    original_protocol = json.loads((root/'data/airfoil_pressure/protocol.json').read_text())
    assert ids == original_protocol['sample_ids']
    methods = ['Uniform','PI','CPG','CA']
    names = {'Uniform':'uniform','PI':'pi_cdcl_tuned','CPG':'cpg_cdcl_budget10','CA':'ca_cdcl_budget10'}
    registry = {r['cohort']:r for r in read_csv(data/'checkpoint_registry.csv')}
    records = read_csv(data/'per_case_records.csv.gz')
    control = read_csv(data/'control_per_case_records.csv.gz')
    curve_rows = read_csv(data/'per_seed_sampling_summary.csv')
    den_rows = read_csv(data/'uniform_denominators.csv')
    training_table = read_csv(data/'training_seed_cdcl_summary.csv')
    checkpoint_table = read_csv(data/'per_training_seed_summary.csv')
    expected = set(itertools.product(range(3),methods,ids,range(3),[.375,.5,.75]))
    actual = [(int(r['training_seed']),r['method'],int(r['sample_id']),int(r['sampling_seed']),float(r['budget'])) for r in records]
    assert len(actual)==len(set(actual))==11880 and set(actual)==expected
    assert len(control)==3960
    assert len(curve_rows)==108 and len(den_rows)==9 and len(training_table)==4 and len(checkpoint_table)==12
    assert len({registry[f'training_seed{s}']['checkpoint_sha256'] for s in range(3)})==3
    assert len({registry[f'training_seed{s}']['head_sha256'] for s in range(3)})==3
    assert len({registry[f'training_seed{s}']['normalizer_sha256'] for s in range(3)})==1
    max_error_identity_difference=0.0
    for r in records+control:
        reg=registry[r['cohort']]
        values=[float(r[k]) for k in ['true_Cd','true_Cl','true_Cd_over_Cl','predicted_Cd','predicted_Cl','predicted_Cd_over_Cl','cdcl_abs_error']]
        assert all(math.isfinite(v) for v in values)
        assert float(r['true_Cl'])>.02 and int(r['sample_id']) in ids and r['status']=='ok'
        assert r['truth_access_during_sampling']=='False'
        assert r['policy_name']==names[r['method']]
        assert r['checkpoint_sha256']==reg['checkpoint_sha256'] and r['normalizer_sha256']==reg['normalizer_sha256']
        assert r['head_sha256']==reg['head_sha256'] if r['method']=='CA' else r['head_sha256']==''
        assert r['reconstruction_id']=='original_index_grid_linear_nearest'
        assert r['interpolation_backend']=='index_grid_linear_nearest'
        assert int(r['coarse_count'])=={'Uniform':0,'PI':0,'CPG':1127,'CA':563}[r['method']]
        ratio=float(r['predicted_Cd'])/float(r['predicted_Cl'])
        assert math.isclose(ratio,float(r['predicted_Cd_over_Cl']),rel_tol=1e-12,abs_tol=1e-12)
        error=abs(float(r['true_Cd_over_Cl'])-ratio)
        diff=abs(error-float(r['cdcl_abs_error']))
        max_error_identity_difference=max(max_error_identity_difference,diff)
        assert math.isclose(error,float(r['cdcl_abs_error']),rel_tol=1e-12,abs_tol=1e-12)
    values={}
    for training_seed in range(3):
        for sampling_seed in range(3):
            curves={method:[statistics.mean(float(r['cdcl_abs_error']) for r in records if
                                           int(r['training_seed'])==training_seed and int(r['sampling_seed'])==sampling_seed and
                                           r['method']==method and float(r['budget'])==budget) for budget in [.375,.5,.75]] for method in methods}
            den=area(curves['Uniform'])
            d=next(r for r in den_rows if int(r['training_seed'])==training_seed and int(r['sampling_seed'])==sampling_seed)
            assert math.isclose(float(d['uniform_area']),den,rel_tol=1e-12,abs_tol=1e-12)
            for method in methods:
                rauec=area(curves[method])/den
                values[training_seed,sampling_seed,method]=rauec
                rr=[r for r in curve_rows if int(r['training_seed'])==training_seed and int(r['sampling_seed'])==sampling_seed and r['method']==method]
                assert len(rr)==3
                for row in rr:
                    i=[.375,.5,.75].index(float(row['budget']))
                    for field,expected_value in [('case_mean_error',curves[method][i]),('uniform_area',den),('method_area',area(curves[method])),('rAUEC',rauec)]:
                        assert math.isclose(float(row[field]),expected_value,rel_tol=1e-12,abs_tol=1e-12)
        for method in methods:
            ss=[values[training_seed,s,method] for s in range(3)]
            row=next(r for r in checkpoint_table if int(r['training_seed'])==training_seed and r['method']==method)
            assert math.isclose(float(row['rAUEC']),statistics.mean(ss),rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(float(row['sampling_sd']),statistics.stdev(ss),rel_tol=1e-12,abs_tol=1e-12)
    for row in training_table:
        means=[statistics.mean(values[t,s,row['method']] for s in range(3)) for t in range(3)]
        for s in range(3):
            assert math.isclose(float(row[f'training_seed_{s}']),means[s],rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(float(row['mean']),statistics.mean(means),rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(float(row['sample_sd']),statistics.stdev(means),rel_tol=1e-12,abs_tol=1e-12)
    original={ (r['method'],int(r['sample_id']),int(r['seed']),float(r['budget_fraction'])):float(r['ratio_abs_error'])
               for r in read_csv(root/'data/airfoil_pressure/cdcl_test_records.csv') }
    control_keys=[(r['method'].lower(),int(r['sample_id']),int(r['sampling_seed']),float(r['budget'])) for r in control]
    assert len(set(control_keys))==3960 and set(control_keys)==set(original)
    max_control_error=max(abs(float(r['cdcl_abs_error'])-original[k]) for r,k in zip(control,control_keys))
    assert max_control_error<=1e-12
    filter_rows=read_csv(data/'case_filter.csv')
    assert len(filter_rows)==200 and [int(r['sample_id']) for r in filter_rows if r['included']=='True']==ids
    assert all((float(r['true_Cl'])>.02)==(r['included']=='True') for r in filter_rows)
    special=read_csv(data/'case_1007_1059_1198.csv')
    assert len(special)==432 and {int(r['sample_id']) for r in special}=={1007,1059,1198}
    result={'passed':True,'training_records':len(records),'control_records':len(control),
            'independent_training_checkpoints':3,'matching_heads':3,'sampling_seeds_per_checkpoint':3,
            'budgets':3,'cases_per_condition':110,'uniform_denominators':9,
            'max_control_per_case_error_difference':max_control_error,
            'max_force_ratio_error_identity_difference':max_error_identity_difference,
            'all_exported_statistics_recomputed':True,'nan_inf_or_missing_records':0,
            'special_case_records':len(special),'mixed_reconstruction':False}
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
