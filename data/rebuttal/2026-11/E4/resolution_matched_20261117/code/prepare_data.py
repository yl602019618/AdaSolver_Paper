"""Build native resolution arrays from real421² fields with an explicit split."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import scipy.io
import torch

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
SOURCE=Path('/mnt/ai-material-data/zhijun/CarD/Neural_Operator_NACA/data/darcy')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text())
    split=json.loads((OUT/'split_manifest.json').read_text())
    assert split['training_ids']==list(range(900)) and split['validation_ids']==list(range(900,1000))
    assert split['test_source_rows']==list(range(200))
    final=OUT/'data/prepared.json'
    if final.exists():
        old=json.loads(final.read_text());assert old['protocol_sha256']==sha(OUT/'protocol.json')
        print('Data preparation already complete');return
    started=time.perf_counter();sources=[];probes=[]
    for source_kind,filename in [('smooth1','piececonst_r421_N1024_smooth1.mat'),('smooth2','piececonst_r421_N1024_smooth2.mat')]:
        path=SOURCE/filename;digest=sha(path)
        for variable,suffix in [('coeff','coeff'),('sol','solution')]:
            raw=scipy.io.loadmat(path,variable_names=[variable])[variable]
            assert raw.shape==(1024,421,421)
            cases=[('train',0,900),('validation',900,1000)] if source_kind=='smooth1' else [('test',0,200)]
            for n in protocol['resolutions']:
                stride=protocol['observation_strides'][str(n)];directory=OUT/'data'/f'r{n}';directory.mkdir(exist_ok=True)
                source_axis=np.arange(0,421,stride);assert len(source_axis)==n and source_axis[-1]==420
                for label,begin,end in cases:
                    values=np.ascontiguousarray(raw[begin:end,::stride,::stride].reshape(end-begin,n*n,1),dtype=np.float32)
                    assert values.shape==(end-begin,n*n,1) and np.isfinite(values).all()
                    target=directory/f'{label}_{suffix}.npy';temp=target.with_suffix('.tmp.npy');np.save(temp,values);temp.replace(target)
                    for case_local in [0,(end-begin)//2,end-begin-1]:
                        for row,col in [(0,0),(1,1),(n//3,n//2),(n-1,n-1)]:
                            observed=values[case_local,row*n+col,0]
                            expected=np.float32(raw[begin+case_local,source_axis[row],source_axis[col]])
                            assert observed==expected
                            probes.append(dict(resolution=n,split=label,variable=variable,source_row=begin+case_local,
                                source_node=[int(source_axis[row]),int(source_axis[col])],native_node=[row,col],value=float(observed),exact_source_match=True))
            del raw
            print(source_kind,variable,'extracted all native resolutions',flush=True)
        sources.append(dict(source=source_kind,path=str(path),sha256=digest,raw_shape=[1024,421,421]))
    resolutions=[]
    for n in protocol['resolutions']:
        directory=OUT/'data'/f'r{n}';norm={}
        for name,label in [('x','coeff'),('y','solution')]:
            value=torch.from_numpy(np.load(directory/f'train_{label}.npy'))
            assert value.shape==(900,n*n,1)
            norm[name+'_mean']=value.mean(dim=(0,1),keepdim=True).numpy()
            norm[name+'_std']=(value.std(dim=(0,1),keepdim=True)+1e-8).numpy()
        np.savez(directory/'normalizer.npz',**norm,training_ids=np.arange(900,dtype=np.int64),resolution=np.array(n))
        axis=np.linspace(0,1,n);x,y=np.meshgrid(axis,axis)
        positions=np.column_stack((x.ravel(),y.ravel())).astype(np.float32)
        np.save(directory/'positions.npy',positions);np.save(directory/'reference_coords.npy',positions.astype(np.float64))
        files={}
        for path in sorted(directory.glob('*.npy')):
            array=np.load(path,mmap_mode='r');files[path.name]=dict(sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype),bytes=path.stat().st_size)
        manifest=dict(status='prepared',resolution=n,native_points=n*n,stride=protocol['observation_strides'][str(n)],
            training_ids=split['training_ids'],validation_ids=split['validation_ids'],test_ids=split['test_ids'],
            normalizer_fit_ids=split['training_ids'],normalizer_fit_files=['train_coeff.npy','train_solution.npy'],
            normalizer_sha256=sha(directory/'normalizer.npz'),normalizer_statistics={k:v.tolist() for k,v in norm.items()},
            files=files,source_case_ranges={'train':['smooth1',0,900],'validation':['smooth1',900,1000],'test':['smooth2',0,200]},
            data_construction='Direct original421-grid nodes; no interpolation or low-resolution reconstruction',
            coordinates='np.meshgridxy;FP32 input, identical values promoted toFP64 for reconstruction',
            split_manifest_sha256=sha(OUT/'split_manifest.json'),protocol_sha256=sha(OUT/'protocol.json'))
        save_json(directory/'data_manifest.json',manifest);resolutions.append(manifest)
    save_json(OUT/'config/source_node_checks.json',dict(passed=True,count=len(probes),probes=probes,
        scope='Direct comparison of saved native nodes with original421-grid source arrays; no synthetic upsampling'))
    save_json(final,dict(prepared=True,protocol_sha256=sha(OUT/'protocol.json'),split_manifest_sha256=sha(OUT/'split_manifest.json'),
        sources=sources,resolutions=resolutions,preparation_seconds=time.perf_counter()-started,
        common211_truth='data/r211/test_solution.npy; same exact physical test cases',
        shapes_verified=True,training_only_normalizers=True))
    print(json.dumps(dict(prepared=True,resolutions=protocol['resolutions'],points=[n*n for n in protocol['resolutions']],seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
