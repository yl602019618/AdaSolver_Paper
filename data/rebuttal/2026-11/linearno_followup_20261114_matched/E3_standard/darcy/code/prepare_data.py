"""Extract the same real85-grid observations and fit only the locked900 cases."""
from pathlib import Path
import json,time
import numpy as np
import scipy.io
import torch
from common import OUT,FOLLOWUP,write,digest

source=Path('/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/darcy')
split=json.loads((FOLLOWUP/'standard_benchmarks/split_manifest.json').read_text())
train=split['training_ids'];validation=split['validation_ids']
assert len(train)==900 and len(validation)==100 and not set(train)&set(validation)
output=OUT/'data';output.mkdir(exist_ok=True)
records=[];started=time.perf_counter()
for kind,filename,count in [('pool','piececonst_r421_N1024_smooth1.mat',1000),('test','piececonst_r421_N1024_smooth2.mat',200)]:
    p=source/filename
    raw=scipy.io.loadmat(p,variable_names=['coeff','sol'])
    assert raw['coeff'].shape==raw['sol'].shape==(1024,421,421)
    for key,label in [('coeff','coeff'),('sol','solution')]:
        field=np.ascontiguousarray(raw[key][:count,::5,::5].reshape(count,7225,1),dtype=np.float32)
        assert np.isfinite(field).all()
        np.save(output/f'{kind}_{label}.npy',field)
    records.append(dict(file=str(p),source_shape=list(raw['coeff'].shape),source_sha256=digest(p),observed_grid=[85,85],stride=5,retained_endpoints=True))
    del raw
    print('extracted',kind,flush=True)
xx=torch.from_numpy(np.load(output/'pool_coeff.npy')[train]);yy=torch.from_numpy(np.load(output/'pool_solution.npy')[train])
normalizer={}
for name,values in [('x',xx),('y',yy)]:
    normalizer[name+'_mean']=values.mean(dim=(0,1),keepdim=True).numpy()
    normalizer[name+'_std']=(values.std(dim=(0,1),keepdim=True)+1e-8).numpy()
np.savez(output/'normalizer.npz',**normalizer)
axis=np.linspace(0,1,85);gx,gy=np.meshgrid(axis,axis)
pos=np.column_stack((gx.ravel(),gy.ravel())).astype(np.float32)
np.save(output/'positions.npy',pos);np.save(output/'reference_coordinates.npy',pos.astype(np.float64))
write(OUT/'config/data_manifest.json',dict(status='prepared',source=records,training_ids=train,validation_ids=validation,
    test_ids=split['darcy']['test_ids'],test_source_rows=split['darcy']['test_raw_rows'],normalizer_fit_ids=train,
    normalizer_sha256=digest(output/'normalizer.npz'),normalizer_statistics={k:v.tolist() for k,v in normalizer.items()},
    shared_split_sha256=digest(FOLLOWUP/'standard_benchmarks/split_manifest.json'),
    coordinate_rule='numpy meshgrid xy [0,1], castfloat32 for model; promote same values tofloat64 for legacy reconstruction',
    history='smooth2 test previously viewed historically; never used for checkpoint/head/policy selection in this new cohort',
    wall_seconds=time.perf_counter()-started))
print('prepared900-onlynormalizer',normalizer,flush=True)
