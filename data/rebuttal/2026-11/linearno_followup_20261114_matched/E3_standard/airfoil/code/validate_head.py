"""Own-backbone/normalizer and supervised train-only CA feature-bank acceptance."""
from common import *
from adasolver.inference.ca.airfoil2d import PressureMLP
import argparse


def validate(architecture):
    operator=read(OUT/f'checkpoints/{architecture}/manifest.json')
    path=OUT/f'checkpoints/{architecture}/head/manifest.json';head=read(path)
    bank_path=Path(head['dataset_manifest']);bank=read(bank_path);data=Data()
    assert operator['execution_status']==head['execution_status']=='completed'
    assert head['operator_sha256']==operator['checkpoint_sha256']==sha(operator['checkpoint'])
    assert head['normalizer_sha256']==operator['normalizer_sha256']==sha(operator['normalizer'])==bank['normalizer_sha256']
    assert bank['operator_sha256']==operator['checkpoint_sha256']
    assert head['feature_tap']==bank['feature_tap']=='frozen_block0_on_S0'
    assert head['target_channel']==3 and head['target_name']=='pressure' and head['target']=='zscore_pressure'
    assert head['operator_frozen'] and head['seed']==0 and head['epochs_requested']==head['epochs_completed']==200
    assert head['architecture_layers']==[128,8,32,1] and head['feature_dim']==8 and head['parameter_count']==1353
    assert head['data_split']['head_train']==bank['train_ids']==data.splits['training_ids']
    assert head['data_split']['head_validation']==bank['validation_ids']==data.splits['validation_ids']
    assert not set(bank['train_ids']+bank['validation_ids']) & set(data.splits['airfoil']['primary_test_ids'])
    assert head['target_mean']==bank['target_mean']==float(data.mean)
    assert head['target_std']==bank['target_std']==float(data.std)
    weights=path.parent/'pressure_d8_best.pt';assert sha(weights)==head['head_sha256']
    model=PressureMLP(8);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    history=head['training_history'];assert len(history)==200 and [r['epoch'] for r in history]==list(range(200))
    best=float('inf');best_epoch=None
    for row in history:
        assert np.isfinite([row['train_huber'],row['validation_huber']]).all()
        if row['validation_huber']<best-1e-7:best=row['validation_huber'];best_epoch=row['epoch']
    assert best==head['best_validation_huber'] and best_epoch==head['best_epoch']
    verified=0;shard_checks=[];mandatory=np.arange(50,171)*51
    for shard in bank['shards']:
        ids=bank['train_ids'] if shard['split']=='train' else bank['validation_ids']
        expected=[sid for sid in ids for _ in range(2)]
        assert [r['sample_id'] for r in shard['entries']]==expected
        indices=np.load(bank_path.parent/shard['indices'],mmap_mode='r')
        targets=np.load(bank_path.parent/shard['target'],mmap_mode='r')
        features=np.load(bank_path.parent/shard['features'],mmap_mode='r')
        assert indices.shape==(len(expected),shard['coarse_count']) and targets.shape==indices.shape+(1,)
        assert features.shape==indices.shape+(128,) and features.dtype==targets.dtype==np.float32
        assert indices.dtype==np.int64 and indices.min()>=0 and indices.max()<11271
        assert np.isfinite(targets).all()
        for row,sid in enumerate(expected):
            chosen=indices[row];assert len(np.unique(chosen))==len(chosen) and np.isin(mandatory,chosen).all()
            _,pressure=data.sample(sid);wanted=(pressure[chosen]-data.mean)/data.std
            if not np.array_equal(targets[row],wanted):raise RuntimeError('feature-bank target does not match train-only normalization and saved selected indices')
            verified+=1
        shard_checks.append(dict(name=shard['name'],records=len(expected),coarse_count=shard['coarse_count'],indices_sha256=sha(bank_path.parent/shard['indices'])))
    assert verified==4000
    receipt=dict(passed=True,architecture=architecture,checkpoint_sha256=operator['checkpoint_sha256'],head_sha256=head['head_sha256'],
        normalizer_sha256=head['normalizer_sha256'],train_cases=900,validation_cases=100,epochs=200,head_parameters=1353,
        verified_bank_rows=verified,head_seed=0,feature_tap=head['feature_tap'],shards=shard_checks,
        selection_rule='native validation Huber improvement greater than1e-7; exactly replayed over all200epochs',
        scope='native strict head state, actual operator/normalizer hash, own train/validation IDs, all selected-index target normalization checks; CA field accuracy separate',
        validator_sha256=sha(__file__),manifest_sha256=sha(path),bank_manifest_sha256=sha(bank_path))
    write(OUT/f'results/{architecture}_head_acceptance.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['t3','linearno'],required=True);args=p.parse_args()
    print(json.dumps(validate(args.architecture),indent=2))
