"""Independent manifest/index/count audit; numerical agreement stays traceable to raw pairs."""
from common import *
import argparse,csv


def validate(architecture):
    gate=read(OUT/'config/evaluation_gate.json');split=read(SPLITS)
    result=read(OUT/f'results/{architecture}_complete_completion.json')
    checkpoint=read(OUT/f'checkpoints/{architecture}/manifest.json')
    expected_methods=['full'] if architecture=='linearno_conv' else ['full','uniform','pi','cpg_beta0','cpg','ca']
    expected={(sid,method,seed,budget) for sid in split['airfoil']['primary_test_ids'] for method in expected_methods
              for seed in ([None] if method=='full' else [0,1,2]) for budget in ([1.] if method=='full' else [.375,.5,.75])}
    records={};pairs=0;maximum_difference=0.;violations=[]
    for source in result['source_files']:
        path=Path(source)
        manifest=read(path.parent/'manifest.json')
        assert manifest['execution_status']=='completed' and not manifest['equivalence_violations']
        assert manifest['reconstruction_gate']['module_sha256']==gate['module_sha256']
        for line in path.open():
            row=json.loads(line);key=(row['case_id'],row['method'],row['sampling_seed'],row['budget_fraction'])
            assert key not in records;records[key]=row
            assert row['checkpoint_sha256']==checkpoint['checkpoint_sha256']
            assert row['normalizer_sha256']==checkpoint['normalizer_sha256']
            if row['method']=='ca':
                head=read(OUT/f'results/{architecture}_head_acceptance.json')
                assert head['passed'] and row['head_sha256']==head['head_sha256']
            for mode in ['old','new']:assert np.isfinite(row[mode]['field_relative_l2']) and row[mode]['field_relative_l2']>=0
            with np.load(path.parent/row['arrays']) as arrays:
                if row['method']=='full':
                    assert arrays['field'].shape==(11271,1) and arrays['field'].dtype==np.float64 and np.isfinite(arrays['field']).all()
                else:
                    ids=arrays['indices'];coarse=arrays['coarse_indices'];pred=arrays['prediction']
                    assert ids.dtype==np.int64 and len(ids)==row['retained_points'] and len(ids)==len(np.unique(ids))
                    assert len(ids)==int(round(11271*row['budget_fraction']))
                    assert ids.min()>=0 and ids.max()<11271 and np.isfinite(pred).all()
                    assert hashlib.sha256(ids.tobytes()).hexdigest()==row['indices_hash']
                    if len(coarse):assert hashlib.sha256(coarse.astype(np.int64).tobytes()).hexdigest()==row['coarse_indices_hash']
                    assert pred.shape==(len(ids),1) and pred.dtype==np.float32
                    assert row['agreement']['passed'] and row['agreement']['elementwise_violations']==0
                    maximum_difference=max(maximum_difference,row['agreement']['max_abs'])
    assert set(records)==expected,(len(records),len(expected))
    for key,row in records.items():
        sid,method,seed,budget=key
        if method!='full':assert (sid,'uniform',seed,budget) in records and (sid,'full',None,1.) in records
        if method=='cpg':
            zero=records[(sid,'cpg_beta0',seed,budget)]
            for field in ['indices_hash','coarse_indices_hash','coarse_prediction_hash']:assert row[field]==zero[field]
            assert row['beta']==1 and zero['beta']==0;pairs+=1
    assert sha(result['metrics_csv'])==result['metrics_sha256']
    with Path(result['metrics_csv']).open() as f:long=list(csv.DictReader(f))
    assert len(long)==len(expected)*8
    validation_groups=['validation_complete'] if architecture=='linearno_conv' else ['validation_preliminary','validation_ca']
    for stage in validation_groups:
        state=read(OUT/f'records/{architecture}/{stage}/manifest.json')
        assert state['execution_status']=='completed' and not state['equivalence_violations']
        assert state['sample_ids']==gate['validation_case_ids']
    receipt=dict(passed=True,architecture=architecture,checkpoint_sha256=checkpoint['checkpoint_sha256'],
        normalizer_sha256=checkpoint['normalizer_sha256'],reconstruction_sha256=gate['module_sha256'],
        unique_prediction_requests=len(records),expected_prediction_requests=len(expected),long_metric_rows=len(long),
        test_cases=len(split['airfoil']['primary_test_ids']),methods=expected_methods,paired_beta_groups=pairs,
        maximum_old_new_absolute_field_difference=maximum_difference,
        acceptance_scope='independent complete matrix/hash/shape/finiteness/paired-indices audit plus raw numerical equivalence receipts; latency separately pending',
        source_files=result['source_files'],metrics_csv=result['metrics_csv'],metrics_sha256=result['metrics_sha256'],validator_sha256=sha(__file__))
    write(OUT/f'results/{architecture}_accuracy_acceptance.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['t3','linearno','linearno_conv'],required=True)
    print(json.dumps(validate(p.parse_args().architecture),indent=2))
