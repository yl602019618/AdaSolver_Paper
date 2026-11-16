"""Extract the predeclared disjoint calibration cases; preserve E4 test data."""
from pathlib import Path
import csv,hashlib,json,os,shutil,time
import numpy as np
import scipy.io

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());cache=OUT/'data_cache'
    if (cache/'prepared.json').exists():
        print('Prepared data already exist');return
    assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256']
    assert sha(protocol['normalizer_file'])==protocol['normalizer_id']
    started=time.perf_counter();digest=sha(protocol['source_mat'])
    assert digest==protocol['source_mat_sha256'],'Source MAT changed'
    for name in ['coeff','sol']:
        values=scipy.io.loadmat(protocol['source_mat'],variable_names=[name])[name]
        assert values.shape==(1024,421,421)
        selected=np.asarray(values[400:500],dtype=np.float32)
        np.save(cache/f'validation_{name}_421.npy',selected)
        del values,selected
        source=ROOT/f'rebuttal_20261112/experiments/E4_darcy_resolution/records/data_cache/{name}_421.npy'
        target=cache/f'test_{name}_421.npy'
        if not target.exists():target.symlink_to(source)
        print('Prepared',name,flush=True)
    original={}
    source_dir=ROOT/'rebuttal_20261112/experiments/E4_darcy_resolution/records'
    for part in ['part0.jsonl','part1.jsonl']:
        with (source_dir/part).open() as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('status')!='ok' or row['method'] not in ['full','uniform','pi']:continue
                n=row['observation_resolution'];k=row['retained_points']
                if row['method']!='full' and k not in {int(round(n*n*b)) for b in protocol['budgets']}:continue
                original[row['request_id']]={key:row[key] for key in ['field_relative_l2_common','selected_indices_sha256','checkpoint_sha256']}
    assert len(original)==11400
    (cache/'legacy_reference.json').write_text(json.dumps(original)+'\n')
    sources={}
    for relative in ['adasolver/models/checkpoint.py','adasolver/models/transolver3.py','adasolver/layers/compensation.py','adasolver/inference/pi/darcy.py','adasolver/inference/uniform/__init__.py','adasolver/utils/seed.py']:
        path=ROOT/'AdaSolver_official'/relative
        if path.exists():
            target=OUT/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);sources[relative]=sha(path)
    result=dict(prepared=True,protocol_sha256=sha(OUT/'protocol.json'),source_mat_sha256=digest,
        calibration_source_rows=protocol['calibration_source_rows'],test_source_rows=protocol['test_source_rows'],
        calibration_disjoint_from_test=True,checkpoint_sha256=protocol['checkpoint_sha256'],normalizer_id=protocol['normalizer_id'],
        files={p.name:dict(path=str(p.resolve()),shape=list(np.load(p,mmap_mode='r').shape),dtype=str(np.load(p,mmap_mode='r').dtype)) for p in cache.glob('*_421.npy')},
        legacy_reference_requests=len(original),source_code_sha256=sources,preparation_seconds=time.perf_counter()-started)
    (cache/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
