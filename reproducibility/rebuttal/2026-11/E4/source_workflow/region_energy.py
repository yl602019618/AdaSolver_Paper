"""Decompose existing Full errors using cached truth and recorded region norms.

This is mean squared relative-error analysis, not a replacement of the paper's
mean relative-L2 metric. Original FP32 truth-norm rounding is retained.
"""
from pathlib import Path
import json,csv
from collections import defaultdict
import numpy as np
from scipy.ndimage import distance_transform_edt

OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'rebuttal_20261112/experiments/E4_darcy_resolution'


def main():
    full={}
    for name in ['part0.jsonl','part1.jsonl']:
        with (SOURCE/'records'/name).open() as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('method')=='full' and row.get('status')=='ok':
                    full[row['case_id'],row['observation_resolution']]=row
    assert len(full)==600
    cache=SOURCE/'records/data_cache'
    coeff=np.load(cache/'coeff_421.npy',mmap_mode='r');sol=np.load(cache/'sol_421.npy',mmap_mode='r')
    x,y=np.meshgrid(np.linspace(0,1,211),np.linspace(0,1,211))
    coords=np.column_stack((x.ravel(),y.ravel())).astype(np.float32).astype(np.float64)
    boundary=np.min(np.column_stack((coords,1-coords)),axis=1)<=.02
    rows=[];inputs=[];checks=[];totals=defaultdict(list)
    for sid in range(1000,1200):
        a=np.asarray(coeff[sid-1000,::2,::2]);u=np.asarray(sol[sid-1000,::2,::2]).ravel()
        interface=np.zeros_like(a,dtype=bool)
        interface[1:]|=np.abs(a[1:]-a[:-1])>1e-6;interface[:-1]|=np.abs(a[1:]-a[:-1])>1e-6
        interface[:,1:]|=np.abs(a[:,1:]-a[:,:-1])>1e-6;interface[:,:-1]|=np.abs(a[:,1:]-a[:,:-1])>1e-6
        near=(distance_transform_edt(~interface,sampling=1/210).ravel()<=.02)&~boundary
        regions={'boundary':boundary,'interface':near,'smooth':~boundary&~near}
        full_norm=float(np.linalg.norm(u));denominator=full_norm**2
        region_norms={name:float(np.linalg.norm(u[mask])) for name,mask in regions.items()}
        weights={name:region_norms[name]**2/denominator for name in regions}
        for n in [85,141,211]:
            record=full[sid,n];pieces=[]
            direct=record['field_relative_l2_common']**2
            for name,mask in regions.items():
                assert record['regions'][name]['evaluation_points']==int(mask.sum())
                contribution=record['regions'][name]['relative_l2']**2*weights[name]
                pieces.append(contribution)
                rows.append(dict(case_id=sid,observation_resolution=n,region=name,
                    truth_energy_fraction=weights[name],normalized_error_energy=contribution,
                    full_squared_relative_l2=direct))
                inputs.append(dict(case_id=sid,observation_resolution=n,region=name,
                    region_relative_l2=record['regions'][name]['relative_l2'],
                    truth_region_norm=region_norms[name],truth_full_norm=full_norm,
                    full_relative_l2=record['field_relative_l2_common']))
            rebuilt=sum(pieces)
            np.testing.assert_allclose(rebuilt,direct,rtol=3e-6,atol=1e-12)
            checks.append(abs(rebuilt-direct)/max(direct,1e-30));totals[n].append(direct)
    summaries=[]
    for n in [85,141,211]:
        for region in ['boundary','interface','smooth']:
            current=[r for r in rows if r['observation_resolution']==n and r['region']==region]
            base=[r for r in rows if r['observation_resolution']==85 and r['region']==region]
            energy=float(np.mean([r['normalized_error_energy'] for r in current]));baseline=float(np.mean([r['normalized_error_energy'] for r in base]))
            increase=float(np.mean(totals[n])-np.mean(totals[85]))
            summaries.append(dict(observation_resolution=n,region=region,
                mean_normalized_squared_error=energy,share_of_mean_squared_relative_error=energy/float(np.mean(totals[n])),
                share_of_increase_vs85=None if n==85 else (energy-baseline)/increase,
                mean_truth_energy_fraction=float(np.mean([r['truth_energy_fraction'] for r in current]))))
    for name,data in [('region_error_energy.csv',summaries),('region_error_energy_per_case.csv',rows),('region_energy_inputs.csv',inputs)]:
        with (OUT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,list(data[0]));writer.writeheader();writer.writerows(data)
    proof=dict(passed=True,cases=200,full_predictions=600,
        definition='Mean over cases of ||prediction-truth||_region^2 / ||truth||_full^2. Region contributions sum to mean squared relative L2, not mean relative L2.',
        derivation='Stored region relative norms squared, multiplied by same-case cached truth region/full energy ratio.',
        rounding='Original FP32 cached truth norms. Residual expected from stored float32/float64 metric arithmetic; tolerance fixed before execution rtol3e-6 atol1e-12.',
        maximum_relative_reconstruction_residual=max(checks),summary=summaries)
    (OUT/'region_energy_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
