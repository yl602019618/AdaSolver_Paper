"""Derive Darcy budget/absolute/rAUEC tables from complete raw metric rows."""
from __future__ import annotations
import csv,json,collections
import numpy as np
from common import OUT,read,write,digest


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    all_rows=[];checkpoints=[]
    lock=read(OUT/'config/evaluation_lock.json')
    for key in ['transolver3','linearno','linearno_conv']:
        directory=OUT/'records'/key;manifest=read(directory/'run_manifest.json')
        if manifest['execution_status']!='completed':raise RuntimeError('accuracy matrix not complete: '+key)
        rows=list(csv.DictReader((directory/'metrics_records.csv').open()));expected=400 if key=='linearno_conv' else 18400
        assert len(rows)==expected
        identity=lambda r:(r['backbone'],r['case_id'],r['sampling_seed'],r['method'],r['beta'],r['budget_fraction'],r['reconstruction_id'])
        assert len({identity(r) for r in rows})==len(rows)
        assert all(r['status']=='ok' and np.isfinite(float(r['value'])) for r in rows)
        all_rows.extend(rows)
        cm=read(OUT/'checkpoints'/key/'manifest.json')
        assert cm['execution_status']=='completed' and cm['epochs_completed']==1000 and cm['total_optimization_steps']==225000
        assert digest(OUT/'checkpoints'/key/'best.pt')==cm['checkpoint_sha256']
        assert digest(OUT/'data/normalizer.npz')==cm['normalizer_sha256']
        assert all(r['checkpoint_id']==cm['checkpoint_sha256'] and r['normalizer_id']==cm['normalizer_sha256']
                   and r['training_seed']=='0' and r['metric_target']=='field' and r['metric']=='field_relative_l2'
                   and r['test_population']=='historical_smooth2_200' for r in rows)
        actual={(int(r['case_id']),r['method'],float(r['budget_fraction']),r['sampling_seed'],r['reconstruction_id']) for r in rows}
        expected_ids={(sid,'full',1.,'',rid) for sid in lock['test_case_ids'] for rid in [lock['old_reconstruction_id'],lock['new_reconstruction_id']]}
        if key!='linearno_conv':
            expected_ids.update((sid,method,budget,str(seed),rid) for sid in lock['test_case_ids']
                for method in ['uniform','pi','cpg_beta0','cpg','ca'] for budget in [.375,.5,.75] for seed in [0,1,2]
                for rid in [lock['old_reconstruction_id'],lock['new_reconstruction_id']])
        assert actual==expected_ids
        for row in rows:
            beta=None if row['method'] in ['full','uniform'] else .5 if row['method']=='pi' else 0. if row['method']=='cpg_beta0' else 1.
            assert (None if row['beta']=='' else float(row['beta']))==beta
        checkpoints.append(dict(dataset='darcy',backbone=rows[0]['backbone'],model_class=cm['model_class'],model_variant=cm['variant'],
            checkpoint_id=cm['checkpoint_sha256'],normalizer_id=cm['normalizer_sha256'],training_seed=0,training_cases=900,validation_cases=100,
            epochs=1000,updates=225000,parameters=cm['parameter_count'],training_seconds=cm['training_wall_seconds'],
            best_validation=cm['best_validation_relative_l2'],checkpoint_path=cm['checkpoint'],identity_verified=True,
            training_completed=True,accuracy_matrix_completed=True,latency_acceptance='pending',
            verification_scope='training completion and byte identity; numeric independence and latency have separate receipts'))
    groups=collections.defaultdict(list)
    for row in all_rows:groups[row['backbone'],row['reconstruction_id'],row['method'],float(row['budget_fraction'])].append(row)
    budget_rows=[]
    for (backbone,recon,method,budget),rows in groups.items():
        values=np.asarray([float(r['value']) for r in rows]);full=groups[(backbone,recon,'full',1.)]
        full_mean=float(np.mean([float(r['value']) for r in full]))
        seeds=sorted({r['sampling_seed'] for r in rows if r['sampling_seed']!=''})
        means=[float(np.mean([float(r['value']) for r in rows if r['sampling_seed']==seed])) for seed in seeds]
        budget_rows.append(dict(dataset='darcy',backbone=backbone,reconstruction_id=recon,method=method,
            beta=rows[0]['beta'],budget_fraction=budget,case_count=len({r['case_id'] for r in rows}),
            sampling_seed_count=len(seeds),mean_error=float(values.mean()),sampling_sd=float(np.std(means,ddof=1)) if len(means)>1 else None,
            same_checkpoint_full_error=full_mean,error_over_full=float(values.mean())/full_mean,
            checkpoint_id=rows[0]['checkpoint_id'],training_seed=0,test_population='historical_smooth2_200',
            normalizer_id=rows[0]['normalizer_id'],metric_target='field',metric='field_relative_l2',unit='ratio',
            retained_points=int(rows[0]['retained_points'])))
        assert budget_rows[-1]['case_count']==200
    rauec=[];seed_details=[]
    for backbone in sorted({r['backbone'] for r in all_rows if r['method']!='full'}):
        for recon in sorted({r['reconstruction_id'] for r in all_rows}):
            for method in ['uniform','pi','cpg_beta0','cpg','ca']:
                ratios=[]
                for seed in ['0','1','2']:
                    curves=[]
                    for m in [method,'uniform']:
                        curve=[np.mean([float(r['value']) for r in groups[backbone,recon,m,b] if r['sampling_seed']==seed]) for b in [.375,.5,.75]]
                        curves.append(curve)
                    den=np.trapezoid(curves[1],[.375,.5,.75]);assert den>0
                    ratios.append(float(np.trapezoid(curves[0],[.375,.5,.75])/den))
                    seed_details.append(dict(dataset='darcy',backbone=backbone,reconstruction_id=recon,method=method,sampling_seed=int(seed),
                        checkpoint_id=groups[backbone,recon,method,.375][0]['checkpoint_id'],
                        mean_error_0375=curves[0][0],mean_error_050=curves[0][1],mean_error_075=curves[0][2],
                        uniform_mean_error_0375=curves[1][0],uniform_mean_error_050=curves[1][1],uniform_mean_error_075=curves[1][2],
                        method_auec=float(np.trapezoid(curves[0],[.375,.5,.75])),uniform_auec=float(den),rauec=ratios[-1]))
                rauec.append(dict(dataset='darcy',backbone=backbone,reconstruction_id=recon,method=method,
                    beta=None if method=='uniform' else .5 if method=='pi' else 0. if method=='cpg_beta0' else 1.,
                    metric_target='field',test_population='historical_smooth2_200',training_seed=0,
                    checkpoint_id=groups[backbone,recon,method,.375][0]['checkpoint_id'],
                    normalizer_id=groups[backbone,recon,method,.375][0]['normalizer_id'],metric='field_relative_l2',unit='relative_area_ratio',
                    seed0=ratios[0],seed1=ratios[1],seed2=ratios[2],mean=float(np.mean(ratios)),sampling_sd=float(np.std(ratios,ddof=1)),
                    sampling_seeds=3,normalization='per-seed AUEC(method)/AUEC(matched Uniform), then mean/sampleSD'))
    csv_write(OUT/'results/backbone_budget_metrics.csv',budget_rows)
    csv_write(OUT/'results/backbone_absolute_errors.csv',budget_rows)
    csv_write(OUT/'results/backbone_rauec.csv',rauec)
    csv_write(OUT/'results/rauec_seed_details.csv',seed_details)
    csv_write(OUT/'results/checkpoint_registry.csv',checkpoints)
    csv_write(OUT/'results/metrics_records.csv',all_rows)
    assert len(all_rows)==37200 and len(budget_rows)==66 and len(rauec)==20
    write(OUT/'results/accuracy_completion.json',dict(execution_status='completed',evidence_status='pending_independent_aggregation',
        total_rows=len(all_rows),physical_requests=18600,budget_rows=len(budget_rows),rauec_rows=len(rauec),
        conv_sampling='N/A: native convolution requires85x85; Full-only reference retained',
        mean_before_area=False,per_case_ratio=False,full_in_rauec=False,
        input_records={key:digest(OUT/'records'/key/'metrics_records.csv') for key in ['transolver3','linearno','linearno_conv']}))
    print(json.dumps(dict(rows=len(all_rows),budget_rows=len(budget_rows),rauec_rows=len(rauec))))


if __name__=='__main__':main()
