"""Freeze twelve validation predictions and profile native CPU reconstruction.

This is a CPU stage diagnostic, not formal GPU end-to-end timing or final
new-cohort standard-benchmark accuracy. All checkpoints are already frozen.
"""
from __future__ import annotations
import hashlib,json,time,sys,os
from pathlib import Path
import numpy as np
import torch
from car_request import OUT,CURRENT,E3,load_backbone,CarRequest,configuration_bank
from profile_native_reconstruction import profile_fixture
from adasolver.models.checkpoint import load_darcy_operator,load_airfoil2d_operator,sha256_file
from adasolver.data.pipe import UnitTransformer
from adasolver.data.airfoil2d_runtime import AirfoilRuntimeData
from adasolver.utils.interpolation import index_grid


def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=OUT/'records/reconstruction_microprofile';root.mkdir(parents=True,exist_ok=True)
    fixture_dir=root/'fixtures';fixture_dir.mkdir(exist_ok=True)
    protocol=json.loads((OUT/'config/real_validation_protocol.json').read_text());ids=protocol['validation_cases']
    frozen=[];records=[]
    def emit(dataset,case,arch,selected,values,coords,checkpoint,mask=None):
        identity=hashlib.sha256((dataset+str(case)+arch).encode()).hexdigest()[:16];path=fixture_dir/(identity+'.npz')
        data=dict(selected=selected,values=values,reference_coordinates=coords)
        if mask is not None:data['surface_mask']=mask
        if path.exists():
            old=np.load(path)
            for k,v in data.items():np.testing.assert_array_equal(old[k],v)
        else:np.savez_compressed(path,**data)
        meta=dict(fixture_id=identity,dataset=dataset,case_id=case,backbone=arch,checkpoint_sha256=checkpoint,
            budget_fraction=.5,sampling_seed=0,selection='validation Uniform',selected_indices_sha256=digest(np.asarray(selected,np.int64)),
            prediction_sha256=digest(values),reference_coordinates_sha256=digest(coords),reference_dtype=str(coords.dtype),
            prediction_dtype=str(values.dtype),output_dtype='float64',prediction_device='CPU',fixture_file=str(path.relative_to(OUT)),
            evidence_scope='CPU reconstruction stage diagnostic; not new-cohort accuracy or GPU timing')
        stages,check=profile_fixture(dataset,coords,selected,values,mask)
        records.extend(dict(**meta,**r) for r in stages);frozen.append(dict(**meta,native_stage_equivalence=check))
        (root/'stages.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (root/'fixtures.json').write_text(json.dumps(frozen,indent=2)+'\n')
        print(dataset,arch,case,'complete',len(frozen),'/12',flush=True)
    for arch in ['Transolver3','LinearNO']:
        runtime,model,context,_=load_backbone(arch,'cpu')
        for case in ids['shapenet_car']:
            item=runtime.sample(case);req=CarRequest(model,context,item,runtime.output_normalizer,configuration_bank()['uniform'],.5,0,device='cpu')
            req.capture_values=True;req('old','A')
            emit('shapenet_car',case,arch,req.last_indices,req.last_physical_values,req.coords,runtime.metadata['checkpoint_sha256'],req.mask)
        del runtime,model
    cache=CURRENT/'linearno_followup_20261114/standard_benchmarks/darcy/data'
    coeff=np.load(cache/'pool_coeff.npy',mmap_mode='r').reshape(1000,-1,1)
    pos=torch.from_numpy(np.load(cache/'positions.npy'));coords=np.load(cache/'reference_coordinates.npy')
    xn=UnitTransformer(torch.from_numpy(np.array(coeff)))
    checkpoint=CURRENT/'AdaSolver_official/checkpoints/darcy/darcy_transolver3_fullmesh_nmin100_s0_best.pt'
    manifest=checkpoint.with_name(checkpoint.name.replace('_best.pt','_manifest.json'));model=load_darcy_operator(checkpoint,manifest,'cpu');model.requires_grad_(False)
    yn=np.load(checkpoint.with_name(checkpoint.name.replace('_best.pt','_ynorm.npz')))
    for case in ids['darcy']:
        selected=np.random.default_rng(np.random.SeedSequence([20261114,int(case),0,5000])).choice(len(coords),round(.5*len(coords)),False)
        fx=xn.encode(torch.from_numpy(np.array(coeff[case:case+1])))[0]
        pred=model(pos[selected][None],fx[selected][None])[0].numpy()
        values=pred*np.asarray(yn['std']).reshape(1,1)+np.asarray(yn['mean']).reshape(1,1)
        emit('darcy',case,'published_Transolver3',selected,values,coords,sha256_file(checkpoint))
    del model,xn,coeff
    checkpoint=CURRENT/'AdaSolver_official/checkpoints/airfoil2d/airfoil2d_transolver3_pressure_nmin040_best.pt'
    manifest=checkpoint.with_name(checkpoint.name.replace('_best.pt','_manifest.json'));model=load_airfoil2d_operator(checkpoint,manifest,'cpu');model.requires_grad_(False)
    air=AirfoilRuntimeData('/mnt/ai-material-data/shuaibinqu/NACA/data/naca',manifest);coords=index_grid((221,51))
    for case in ids['airfoil']:
        physical=air.coords(case);available=np.setdiff1d(np.arange(len(coords)),air.mandatory)
        rng=np.random.default_rng(np.random.SeedSequence([20261114,int(case),0,5000]))
        selected=np.r_[air.mandatory,rng.choice(available,round(.5*len(coords))-len(air.mandatory),False)]
        x=torch.tensor(physical[selected][None]);values=air.decode(model(x,x))[0].numpy()
        emit('airfoil',case,'published_Transolver3',selected,values,coords,sha256_file(checkpoint))
    (OUT/'results/reconstruction_microprofile.json').write_text(json.dumps(dict(status='completed',fixtures=len(frozen),
        stage_observations=len(records),profile_repeats=5,device='CPU',threads=4,pid=os.getpid(),
        accuracy_scope='published checkpoints on fixed validation cases; no new test-set or new-cohort evidence',
        fused_stage='Native SciPy grid search/weights/application is compiled and reported as one interval',
        maximum_staged_native_absolute_difference=max(r['native_stage_equivalence']['max_abs'] for r in frozen)),indent=2)+'\n')


if __name__=='__main__':main()
