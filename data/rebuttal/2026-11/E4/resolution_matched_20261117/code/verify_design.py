"""Independent source, split, normalization and completed-training identity audit."""
from pathlib import Path
import argparse,csv,datetime as dt,hashlib,json,math,sys,time
import numpy as np
import scipy.io

OUT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify_data():
    p=read(OUT/'protocol.json');s=read(OUT/'split_manifest.json');prep=read(OUT/'data/prepared.json')
    assert s['training_ids']==list(range(900)) and s['validation_ids']==list(range(900,1000))
    assert s['test_ids']==list(range(1000,1200)) and s['test_source_rows']==list(range(200))
    assert sha(OUT/'protocol.json')==prep['protocol_sha256']
    rows=[];normalizers=[]
    for source in prep['sources']:
        path=Path(source['path']);assert sha(path)==source['sha256']
        ranges=[('train',0,900),('validation',900,1000)] if source['source']=='smooth1' else [('test',0,200)]
        for variable,suffix in [('coeff','coeff'),('sol','solution')]:
            raw=scipy.io.loadmat(path,variable_names=[variable])[variable]
            assert raw.shape==(1024,421,421)
            for n,stride in [(85,5),(141,3),(211,2)]:
                folder=OUT/f'data/r{n}';m=read(folder/'data_manifest.json')
                for label,start,end in ranges:
                    filename=f'{label}_{suffix}.npy';a=np.load(folder/filename,mmap_mode='r')
                    assert a.shape==(end-start,n*n,1) and a.dtype==np.float32
                    assert sha(folder/filename)==m['files'][filename]['sha256']
                    for begin in range(0,end-start,20):
                        stop=min(end-start,begin+20)
                        expected=raw[start+begin:start+stop,0:421:stride,0:421:stride].astype(np.float32).reshape(stop-begin,n*n,1)
                        assert np.array_equal(a[begin:stop],expected),(n,label,variable,begin)
                    rows.append(dict(resolution=n,split=label,variable=variable,elements=int(a.size),all_elements_exact=True))
            del raw
    for n in [85,141,211]:
        folder=OUT/f'data/r{n}';m=read(folder/'data_manifest.json')
        assert m['training_ids']==list(range(900)) and m['validation_ids']==list(range(900,1000))
        assert m['test_ids']==list(range(1000,1200)) and m['normalizer_fit_ids']==list(range(900))
        assert m['normalizer_sha256']==sha(folder/'normalizer.npz')
        with np.load(folder/'normalizer.npz',allow_pickle=False) as norm:
            assert norm['training_ids'].tolist()==list(range(900)) and norm['resolution'].item()==n
            for key,kind in [('x','coeff'),('y','solution')]:
                a=np.load(folder/f'train_{kind}.npy',mmap_mode='r')
                mean=float(np.mean(a,dtype=np.float64));std=float(np.std(a,dtype=np.float64,ddof=1))+1e-8
                actual=[float(norm[key+'_mean'].item()),float(norm[key+'_std'].item())]
                # FP32 fitting is compared with an independent FP64 reduction.
                np.testing.assert_allclose(actual,[mean,std],rtol=8*np.finfo(np.float32).eps,atol=1e-10)
                normalizers.append(dict(resolution=n,variable=kind,fit_rows=[0,900],actual=actual,
                    independent_fp64=[mean,std],absolute_differences=[abs(actual[0]-mean),abs(actual[1]-std)]))
    result=dict(passed=True,scope='all saved train/validation/test elements versus original421 source plus training-only normalizers',
        protocol_sha256=sha(OUT/'protocol.json'),split_sha256=sha(OUT/'split_manifest.json'),arrays=rows,
        normalizers=normalizers,verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(OUT/'results/data_identity_acceptance.json',result)
    return result


def verify_training(n):
    folder=OUT/f'checkpoints/r{n}';m=read(folder/'manifest.json');protocol=read(OUT/'protocol.json')
    assert m['execution_status']=='completed' and m['epochs_completed']==1000
    assert m['total_optimization_steps']==225000 and m['updates_per_epoch']==225
    assert m['effective_batch_size']==4 and m['training_seed']==0 and m['trained_from_scratch']
    assert m['training_ids']==list(range(900)) and m['validation_ids']==list(range(900,1000))
    assert m['test_used_for_selection'] is False and m['test_opened_by_training'] is False
    assert all('test_' not in Path(x).name for x in m['input_data_access'])
    assert m['train_coefficient_shape']==[900,n*n,1] and m['validation_coefficient_shape']==[100,n*n,1]
    assert m['nfull']==m['nmin_count']==n*n and m['nmin_fraction']==1.
    assert m['actual_shape_evidence']['preprocess_input_shape'][1]==n*n
    assert m['normalizer_sha256']==sha(OUT/f'data/r{n}/normalizer.npz')
    assert m['protocol_sha256']==sha(OUT/'protocol.json') and m['split_manifest_sha256']==sha(OUT/'split_manifest.json')
    assert m['checkpoint_sha256']==sha(folder/'best.pt')
    assert m['initial_checkpoint_sha256']==sha(folder/'initial.pt')
    assert m['last_checkpoint_sha256']==sha(folder/'last.pt')
    for rel,digest in m['source_sha256'].items():assert sha(OUT/'code'/rel)==digest
    for rel,digest in m['native_model_source_sha256'].items():assert sha(OUT/'code'/rel)==digest
    histories=[read(f) for f in sorted((folder/'history').glob('epoch_*.json'))]
    assert len(histories)==1000
    assert [h['epoch'] for h in histories]==list(range(1,1001))
    assert all(math.isfinite(h['validation_field_l2']) for h in histories)
    best=min(histories,key=lambda h:h['validation_field_l2'])
    assert best['epoch']==m['best_epoch'] and best['validation_field_l2']==m['best_validation_relative_l2']
    head_folder=folder/'head';bank_folder=OUT/f'head_banks/r{n}'
    head=read(head_folder/'manifest.json');bank=read(bank_folder/'manifest.json')
    hv=read(OUT/f'results/head_r{n}_validation.json')
    coarse=int(round(n*n*protocol['head']['coarse_fraction']))
    assert head['execution_status']==bank['execution_status']=='completed' and hv['passed']
    assert head['operator_sha256']==bank['operator_sha256']==m['checkpoint_sha256']
    assert head['operator_tensor_sha256']==bank['operator_tensor_sha256']
    assert head['normalizer_sha256']==bank['normalizer_sha256']==m['normalizer_sha256']
    assert head['resolution']==bank['resolution']==n and head['epochs_completed']==head['epochs']==1000
    assert head['nfull']==bank['nfull']==n*n and head['coarse_count']==bank['coarse_count']==coarse
    assert head['total_optimization_steps']==225000
    assert head['architecture']=='TransolverGradientHead' and head['parameter_count']==236049
    for key in ['blocks','hidden_dim','heads','slices','mlp_ratio']:
        assert head[key]==protocol['head'][key]
    settings=head['settings']
    for key,locked in [('epochs','epochs'),('effective_batch','effective_batch_size'),
                       ('learning_rate','lr'),('weight_decay','weight_decay'),('max_grad_norm','max_grad_norm'),
                       ('optimizer','optimizer'),('scheduler','scheduler'),('pct_start','pct_start'),('head_seed','seed')]:
        assert settings[key]==protocol['head'][locked],('Head configuration differs',key)
    assert settings['steps_per_epoch']==225 and settings['total_updates']==225000
    assert head['head_seed']==0 and head['world_size']==protocol['head']['world_size']
    assert head['local_batch_size']*head['world_size']==settings['effective_batch']
    assert head['micro_batch_size']*head['gradient_accumulation_steps']==head['local_batch_size']
    assert head['validation_rule']==settings['selection']=='minimum validation mean per-case vector error norm'
    assert head['training_ids']==list(range(900)) and head['validation_ids']==list(range(900,1000))
    assert head['test_used_for_selection'] is False and head['test_inputs_opened']==[]
    assert bank['test_ids_used']==[] and bank['teacher_grid_shape']==[n,n]
    assert bank['training_ids']==list(range(900)) and bank['validation_ids']==list(range(900,1000))
    assert bank['head_seed']==0 and bank['masks_per_case']==1 and bank['completed_rows']==1000
    assert bank['frozen_operator_unchanged'] and bank['target_normalizer_fit']=='training rows0:900 only'
    assert bank['target']==head['target']=='zscore_log1p_grad_lap'
    assert math.isfinite(bank['target_mean']) and math.isfinite(bank['target_std']) and bank['target_std']>0
    assert head['target_mean']==bank['target_mean'] and head['target_std']==bank['target_std']
    assert head['checkpoint_sha256']==head['head_sha256']==sha(head_folder/'best.pt')
    assert head['initial_checkpoint_sha256']==sha(head_folder/'initial.pt')
    assert head['last_checkpoint_sha256']==sha(head_folder/'last.pt')
    assert head['bank_manifest_sha256']==sha(bank_folder/'manifest.json')
    expected_shapes={'features.npy':[1000,coarse,128],'teacher_predictions.npy':[1000,coarse,1],
        'raw_scores.npy':[1000,coarse],'coarse_indices.npy':[1000,coarse],
        'sample_ids.npy':[1000],'targets.npy':[1000,coarse,1]}
    assert bank['file_shapes']==head['source_bank_shapes']==expected_shapes
    assert set(bank['files'])==set(expected_shapes)
    for name,expected in bank['files'].items():
        assert sha(bank_folder/name)==expected,('Head bank file changed',name)
        assert list(np.load(bank_folder/name,mmap_mode='r',allow_pickle=False).shape)==expected_shapes[name]
    assert bank['sample_ids_sha256']==bank['files']['sample_ids.npy']
    np.testing.assert_array_equal(np.load(bank_folder/'sample_ids.npy',allow_pickle=False),np.arange(1000))
    expected_sources={'train_coeff':'train_coeff.npy','validation_coeff':'validation_coeff.npy',
        'positions':'positions.npy','reference':'reference_coords.npy','normalizer':'normalizer.npz'}
    assert set(bank['accessed_data_files'])==set(bank['accessed_data_sha256'])==set(expected_sources)
    for name,filename in expected_sources.items():
        path=OUT/f'data/r{n}'/filename
        assert Path(bank['accessed_data_files'][name]).resolve()==path.resolve()
        assert bank['accessed_data_sha256'][name]==sha(path)
    for manifest,archive in [(head,head_folder/'source_at_start'),(bank,bank_folder/'source_at_start')]:
        for name,expected in manifest['source_sha256'].items():
            assert sha(OUT/'code'/name)==sha(archive/name)==expected,('Head numerical source changed',name)
    frozen_vendor=read(OUT/'config/source_manifest.json')['files']
    for name in ['inference/ca/head.py','models/transolver3.py','inference/acquisition.py']:
        assert sha(OUT/'code/vendor/adasolver'/name)==frozen_vendor[name],('Native head/teacher source changed',name)
    shapes=head['actual_training_shapes']
    assert shapes and any(row['epoch']==1000 for row in shapes)
    for row in shapes:
        assert row['native_resolution']==[n,n]
        assert row['features']==[head['local_batch_size'],coarse,128]
        assert row['target']==[head['local_batch_size'],coarse,1]
        assert row['micro_prediction']==[head['micro_batch_size'],coarse,1]
        assert len(row['global_sample_ids'])==4 and len(row['local_sample_ids'])==head['local_batch_size']
        assert all(0<=case<900 for case in row['global_sample_ids'])
        assert set(row['local_sample_ids']).issubset(row['global_sample_ids'])
    head_rows=[json.loads(line) for line in (head_folder/'history.jsonl').read_text().splitlines() if line.strip()]
    assert len(head_rows)==1000 and [r['epoch'] for r in head_rows]==list(range(1,1001))
    assert [r['global_step'] for r in head_rows]==[225*i for i in range(1,1001)]
    assert all(math.isfinite(r['validation']) and math.isfinite(r['train_loss']) for r in head_rows)
    hbest=min(head_rows,key=lambda h:h['validation'])
    assert hbest['epoch']==head['best_epoch'] and hbest['validation']==head['best_validation']
    # A passed receipt must still belong to these exact weights and bank files.
    assert hv['resolution']==n and hv['head_sha256']==head['checkpoint_sha256']
    assert hv['operator_sha256']==m['checkpoint_sha256'] and hv['normalizer_sha256']==m['normalizer_sha256']
    assert hv['bank_manifest_sha256']==head['bank_manifest_sha256']
    assert hv['verifier_sha256']==sha(OUT/'code/verify_head.py')
    assert hv['epochs_completed']==1000 and hv['logical_optimizer_steps']==225000 and hv['history_rows']==1000
    assert hv['bank_records']==1000 and hv['training_cases']==900 and hv['validation_cases']==100
    assert hv['test_cases_used_for_training_or_selection']==0
    assert hv['model_parameter_count']==head['parameter_count']
    assert hv['full_native_grid']==[n,n] and hv['coarse_count']==coarse
    assert hv['all_coarse_indices_reproduced'] and hv['all_standardized_targets_reproduced']
    assert {row['case_id'] for row in hv['teacher_spot_checks']}=={0,449,899,900,999}
    assert all(row['native_grid']==[n,n] and row['coarse_count']==coarse and row['maximum_difference']==0.
               for row in hv['teacher_spot_checks'])
    assert hv['bank_reproduction_batch_size']==8
    expected_groups=[list(range(0,8)),list(range(896,904))]
    assert hv['bank_reproduction_case_groups']==expected_groups
    assert hv['reproduction_tolerance']=='abs(actual-expected) <= 1e-6 + 1e-5*abs(expected)'
    expected_reproduction={f'bank_batch{lower}:{upper}_{kind}' for lower,upper in [(0,8),(896,904)]
                           for kind in ['block0','normalized_teacher_prediction','extractor_vs_full_forward']}
    expected_reproduction.update(f'case{case}_{kind}' for case in [0,900]
                                for kind in ['batch1_extractor_vs_full_forward','batch1_head_vs_hook_features'])
    assert {row['name'] for row in hv['feature_and_prediction_reproduction']}==expected_reproduction
    assert len(hv['feature_and_prediction_reproduction'])==10
    assert all(row['violations']==0 and math.isfinite(row['relative_norm'])
               for row in hv['feature_and_prediction_reproduction'])
    # Different GEMM batch shapes are a measured numerical diagnostic, while
    # each execution context must independently reproduce its own feature tap.
    sensitivity=hv['batch_size_sensitivity_diagnostics']
    assert len(sensitivity)==4
    assert {row['name'] for row in sensitivity}=={f'case{case}_batch1_vs_bank8_{kind}'
        for case in [0,900] for kind in ['block0','prediction']}
    assert all(math.isfinite(row['relative_norm']) and math.isfinite(row['max_abs']) for row in sensitivity)
    interfaces=hv['single_request_interface_checks']
    assert len(interfaces)==2 and {row['case_id'] for row in interfaces}=={0,900}
    assert all(row['input_shape']==row['output_shape']==[1,coarse,1]
               and row['native_resolution']==n and row['finite_prediction_and_decoded_score'] for row in interfaces)
    return dict(resolution=n,passed=True,backbone_sha256=m['checkpoint_sha256'],head_sha256=head['checkpoint_sha256'],
        normalizer_sha256=m['normalizer_sha256'],backbone_best_epoch=m['best_epoch'],head_best_epoch=head['best_epoch'],
        initial_state_tensor_sha256=m['initial_state_tensor_sha256'],backbone_epochs=1000,backbone_updates=225000,
        parameter_count=m['parameter_count'],training_wall_seconds=m['training_wall_seconds'],
        head_initial_tensor_sha256=head['initial_tensor_sha256'],head_epochs=1000,head_updates=225000,
        head_parameter_count=head['parameter_count'],head_training_wall_seconds=head['training_seconds'],
        head_bank_wall_seconds=bank['bank_seconds'],head_bank_manifest_sha256=head['bank_manifest_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-only',action='store_true');args=parser.parse_args()
    receipt=OUT/'results/data_identity_acceptance.json'
    if not receipt.exists():verify_data()
    else:assert read(receipt)['passed'] and read(receipt)['protocol_sha256']==sha(OUT/'protocol.json')
    if not args.data_only:
        rows=[verify_training(n) for n in [85,141,211]]
        assert len({r['initial_state_tensor_sha256'] for r in rows})==1
        write(OUT/'results/design_acceptance.json',dict(passed=True,resolutions=rows,
            protocol_sha256=sha(OUT/'protocol.json'),scope='matched split, scratch initialization, native shapes, completed backbone and own-head identity'))
    print('data identity passed' if args.data_only else 'complete design identity passed',flush=True)
