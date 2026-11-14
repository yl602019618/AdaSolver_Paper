from common import *
import platform

def main():
    split=read(SPLITS);train=split['training_ids'];val=split['validation_ids']
    assert len(train)==900 and len(val)==100 and not set(train)&set(val)
    x=np.load(DATA/'NACA_Cylinder_X.npy',mmap_mode='r');y=np.load(DATA/'NACA_Cylinder_Y.npy',mmap_mode='r');q=np.load(DATA/'NACA_Cylinder_Q.npy',mmap_mode='r')
    assert x.shape==y.shape==(2490,221,51) and q.shape==(2490,5,221,51)
    coords=np.stack((x[:1000],y[:1000]),axis=-1).reshape(1000,11271,2).astype(np.float32)
    pressure=np.asarray(q[:1000,3],dtype=np.float32).reshape(1000,11271,1)
    assert np.isfinite(coords).all() and np.isfinite(pressure).all()
    training=pressure[train]
    mean=np.float32(np.mean(training,dtype=np.float64));std=np.float32(np.std(training,dtype=np.float64,ddof=1)+1e-8)
    np.save(OUT/'cache/coords_trainpool.npy',coords);np.save(OUT/'cache/pressure_trainpool.npy',pressure)
    np.savez(OUT/'cache/normalizer.npz',mean=np.array([[[mean]]],dtype=np.float32),std=np.array([[[std]]],dtype=np.float32),training_ids=np.asarray(train,dtype=np.int64))
    record=dict(benchmark='airfoil2d',target='pressure',target_channel=3,shape=[221,51],source_shape=list(q.shape),
                train_ids=train,validation_ids=val,primary_test_ids=split['airfoil']['primary_test_ids'],
                secondary_test_ids=split['airfoil']['secondary_pressure_test_ids'],
                normalization='global pressure channel, fitted on900 training samples only; float64 accumulation saved asfloat32',
                pressure_mean=float(mean),pressure_std=float(std),normalizer_sha256=sha(OUT/'cache/normalizer.npz'),
                split_manifest_sha256=sha(SPLITS),training_input_sha256=sha(OUT/'cache/coords_trainpool.npy'),
                pressure_pool_sha256=sha(OUT/'cache/pressure_trainpool.npy'),
                source_files={name:dict(path=str(DATA/name),bytes=(DATA/name).stat().st_size) for name in ['NACA_Cylinder_X.npy','NACA_Cylinder_Y.npy','NACA_Cylinder_Q.npy']},
                test_used_for_normalization=False,test_used_for_selection=False,
                history='The historical200-case test pool was previously viewed; this is not a never-touched holdout.')
    write(OUT/'config/data_manifest.json',record)
    protocol=dict(protocol_id='airfoil_matched_pressure_20261114_v1',architectures=['t3','linearno'],
                  training_seed=0,head_seed=0,epochs=1000,batch_size=8,steps_per_epoch=113,total_updates=113000,
                  learning_rate=.001,weight_decay=.00001,optimizer='AdamW',scheduler='OneCycleLR',pct_start=.3,
                  max_grad_norm=dict(t3=.1,linearno=1.,linearno_conv=1.),loss='sum of per-case physical-space pressure relativeL2 after decoding',
                  point_schedule=dict(minimum=4508,maximum=11271,inclusive=True,common_K_per_batch=True,
                                      count_distribution='discrete_uniform',selected_points='uniform_without_replacement',
                                      independent_data_rng_seed=1001,independent_point_rng_seed=2001),
                  model_selection='minimum full-point physical relativeL2 on fixed100 validation cases',
                  t3=dict(class_name='adasolver.models.adapter.Transolver3Adapter',hidden=128,layers=8,heads=8,mlp_ratio=2,slices=64,input='rawxy duplicated as pos/fx,4 channels'),
                  linearno=dict(class_name='official Standard_PDE_Benchmark.LinearAttentionNeuralOperator',
                                attention_class='LinearNO_temp',args_model='temp',hidden=128,layers=8,heads=8,
                                mlp_ratio=1,key_dimension=64,key_ratio_semantics='absolute key width',input='rawxy,2 channels',
                                source_commit='3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269'),
                  head=dict(architecture=[128,8,32,1],activation='GELU',epochs=200,patience=201,
                            fractions=[.05,.1],masks_per_geometry=2,batch_points=65536,learning_rate=.001,
                            weight_decay=.00001,max_grad_norm=1.,loss='smooth_l1_beta1',feature_tap='own frozen first block on coarse subset'),
                  extra_lr_search_budget=0,network_execution='native eager float32, TF32 disabled',
                  reconstruction='pending task1 accepted reconstruction ID before formal test/timing',
                  history_differences=['old model fit1000 and selected on200 historical test; new fit900/val100',
                                       'old T3 runner trained normalized target relativeL2; new shared loss uses physical pressure',
                                       'official Airfoil script predictsMach; this requested task predictspressure',
                                       'official conv_temp script cannot accept arbitrary point subsets; point-temp is a distinct native official variant'],
                  split_manifest_sha256=sha(SPLITS),normalizer_sha256=record['normalizer_sha256'],
                  environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda))
    write(OUT/'config/protocol_lock.json',protocol)
    print(json.dumps({k:record[k] for k in ['pressure_mean','pressure_std','normalizer_sha256','split_manifest_sha256']},indent=2))

if __name__=='__main__':main()
