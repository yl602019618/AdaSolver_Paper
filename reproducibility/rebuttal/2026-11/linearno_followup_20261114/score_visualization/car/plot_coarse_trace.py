#!/usr/bin/env python3
"""CSV-only CPG coarse prediction -> raw gradient -> normalized score trace."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LogNorm
from matplotlib.cm import ScalarMappable
from PIL import Image


def read_csv(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    return {key:np.array([float(r[key]) if r[key] else np.nan for r in rows]) for key in rows[0]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--coarse-csv',type=Path,required=True);p.add_argument('--perpoint-csv',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    coarse,full=read_csv(args.coarse_csv),read_csv(args.perpoint_csv)
    xyz=np.column_stack([full['x'],full['y'],full['z']]).astype(np.float32).astype(np.float64)
    coarse_xyz=np.column_stack([coarse['x'],coarse['y'],coarse['z']]).astype(np.float32).astype(np.float64)
    ids=coarse['native_index'].astype(np.int64);order=coarse['coarse_order'].astype(np.int64)
    assert np.array_equal(order,np.arange(len(ids))) and len(ids)==3218 and len(np.unique(ids))==3218
    assert np.array_equal(coarse_xyz,xyz[ids])
    surface=full['surface'].astype(bool);active=coarse['surface'].astype(bool)
    assert np.array_equal(active,surface[ids])
    # Restore the original FP32 predictions before native FP64 score arithmetic.
    pressure=coarse['predicted_pressure'].astype(np.float32).astype(np.float64)
    cp,cv=coarse_xyz[active],pressure[active]
    distance,neighbor=cKDTree(cp).query(cp,k=5)
    local=np.zeros(len(cp),dtype=np.float64)
    for column in range(1,5):
        delta=cv-cv[neighbor[:,column]]
        local+=delta*delta/(distance[:,column]**2+1e-12)
    local/=4
    on_surface=np.maximum(local,0)[cKDTree(cp).query(xyz[surface],k=1)[1]]
    distance,nearest=cKDTree(xyz[surface]).query(xyz,k=1)
    raw=on_surface[nearest]*np.exp(-distance/.1)
    normalized=np.maximum(raw/max(float(np.quantile(raw,.95)),1e-12),.1)
    raw_difference=np.abs(raw-full['raw_score_cpu']);normalized_difference=np.abs(normalized-full['normalized_score_cpu'])
    np.testing.assert_array_equal(raw,full['raw_score_cpu'])
    np.testing.assert_array_equal(normalized,full['normalized_score_cpu'])
    coords=xyz[:,[2,0,1]];bounds=np.array([coords.min(0),coords.max(0)]);aspect=np.ptp(coords,axis=0)
    center=.5*(xyz[:,0].min()+xyz[:,0].max());display=surface|(np.abs(xyz[:,0]-center)<=.03*np.ptp(xyz[:,0]))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'adasolver-car-cpg-coarse-v1','savefig.dpi':300})
    fig=plt.figure(figsize=(11.5,3.6),layout='constrained');panels=[]
    values_full=np.full(len(xyz),np.nan);values_full[ids]=pressure
    specifications=[('A · Coarse predicted pressure (CPU)',ids[active],values_full,'coolwarm','Generated surface pressure'),
                    ('B · Raw gradient acquisition score',np.flatnonzero(display),full['raw_score_cpu'],'viridis','Gradient energy (log scale)'),
                    ('C · q95-normalized score (CPU)',np.flatnonzero(display),full['normalized_score_cpu'],'viridis','Normalized refinement score (log scale)')]
    for i,(title,show,values,cmap,label) in enumerate(specifications):
        ax=fig.add_subplot(1,3,i+1,projection='3d',proj_type='ortho');ax.view_init(elev=18,azim=-62);ax.set(xlim=bounds[:,0],ylim=bounds[:,1],zlim=bounds[:,2]);ax.set_box_aspect(aspect);ax.set_axis_off();ax.set_title(title)
        norm=Normalize(float(values[show].min()),float(values[show].max())) if i==0 else LogNorm(float(values[show][values[show]>0].min()),float(values[show].max()))
        ax.scatter(*coords[show].T,c=values[show],norm=norm,cmap=cmap,s=3 if i==0 else 2.5,depthshade=False,linewidths=0)
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,location='bottom',shrink=.85,pad=.01,label=label)
        panels.append({'title':title,'source_field':['predicted_pressure','raw_score_cpu','normalized_score_cpu'][i],'shown_points':len(show)})
    for extension in ['pdf','svg','png']:
        meta={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None} if extension=='svg' else {}
        fig.savefig(args.output_dir/f'car_cpg_cpu_coarse_trace.{extension}',bbox_inches='tight',pad_inches=.06,metadata=meta)
    plt.close(fig)
    with Image.open(args.output_dir/'car_cpg_cpu_coarse_trace.png') as image:image.load();image_qa={'PNG_decode':'passed','size':image.size,'dpi':image.info.get('dpi')}
    receipt={'status':'VERIFIED_COMPLETE','scope':'CPU coarse-prediction-to-score trace; no new forward and no GPU',
        'coarse_csv_sha256':hashlib.sha256(args.coarse_csv.read_bytes()).hexdigest(),'perpoint_csv_sha256':hashlib.sha256(args.perpoint_csv.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'coarse_ordered_hash':hashlib.sha256(ids.tobytes()).hexdigest(),
        'coarse_points':len(ids),'coarse_surface_points':int(active.sum()),'coordinates_exact':True,
        'raw_score_from_exported_prediction_max_difference':float(raw_difference.max()),'normalized_score_max_difference':float(normalized_difference.max()),
        'raw_and_normalized_score_exact':True,'truth_input':False,'archived_GPU_raw_claim':False,'panels':panels,'render':image_qa}
    (args.output_dir/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
