#!/usr/bin/env python3
"""CSV-only view of the independent CPU CA score-to-prediction diagnostic."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize,PowerNorm
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    with args.csv.open() as f:rows=list(csv.DictReader(f))
    a={key:np.array([float(r[key]) if r[key] else np.nan for r in rows]) for key in rows[0]}
    xyz=np.column_stack([a['x'],a['y'],a['z']]).astype(np.float32);surface=a['surface'].astype(bool);selected=a['selected'].astype(bool);coarse=a['coarse'].astype(bool)
    center=(xyz[:,0].min()+xyz[:,0].max())*.5;display=surface|(np.abs(xyz[:,0]-center)<=.03*np.ptp(xyz[:,0]));coords=xyz[:,[2,0,1]]
    bounds=np.array([coords.min(0),coords.max(0)]);aspect=np.ptp(coords,axis=0)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'svg.fonttype':'none','svg.hashsalt':'adasolver-car-ca-cpu-v1','pdf.fonttype':42,'savefig.dpi':300})
    fig=plt.figure(figsize=(14,7.0),layout='constrained');panels=[]
    specs=[('A · Own-backbone feature summary','coarse_feature_l2',coarse&surface,'viridis','Block0 feature L2 norm (256 channels)'),
           ('B · Own-head raw output','coarse_head_output_raw',coarse&surface,'coolwarm','Standardized pressure proxy'),
           ('C · Raw gradient acquisition score','raw_score',display,'viridis','Raw gradient energy (log scale)'),
           ('D · Normalized acquisition score','normalized_score',display,'viridis','q95-normalized score (log scale)'),
           ('E · Designed target density','target_density',display,'viridis','Target probability mass; sum = 1'),
           ('F · Actual selected points',None,display,None,None),
           ('G · Own complete-field prediction','predicted_surface_pressure',surface,'coolwarm','Predicted native surface pressure'),
           ('H · Own-prediction error diagnostic','surface_pressure_absolute_error',surface,'magma','Absolute pressure error; square-root scale')]
    for i,(title,key,support,cmap,label) in enumerate(specs):
        ax=fig.add_subplot(2,4,i+1,projection='3d',proj_type='ortho');ax.view_init(elev=18,azim=-62);ax.set(xlim=bounds[:,0],ylim=bounds[:,1],zlim=bounds[:,2]);ax.set_box_aspect(aspect);ax.set_axis_off();ax.set_title(title)
        if key is None:
            ids=np.flatnonzero(display&(selected|surface));colors=np.where(coarse[ids],'#D55E00',np.where(selected[ids],'#222222','#d6d6d6'))
            ax.scatter(*coords[ids].T,c=colors,s=2.5,depthshade=False,linewidths=0)
            ax.text2D(.5,.01,'16,093 selected; 3,218 / 3,218 coarse retained',transform=ax.transAxes,ha='center',fontsize=8)
        else:
            ids=np.flatnonzero(support&np.isfinite(a[key]));values=a[key]
            if key in ['raw_score','normalized_score','target_density']:
                positives=values[ids][values[ids]>0];norm=LogNorm(float(positives.min()),float(positives.max()))
            elif key=='surface_pressure_absolute_error':norm=PowerNorm(.5,0,float(np.nanmax(values)))
            else:norm=Normalize(float(np.nanmin(values[ids])),float(np.nanmax(values[ids])))
            ax.scatter(*coords[ids].T,c=values[ids],cmap=cmap,norm=norm,s=3 if key=='coarse_pressure_proxy' else 2.5,depthshade=False,linewidths=0)
            fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,location='bottom',shrink=.83,pad=.015,label=label)
        panels.append({'title':title,'field':key,'support_points':len(ids)})
    for ext in ['pdf','svg','png']:
        meta={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {}
        fig.savefig(args.output_dir/f'car_ca_cpu_complete_chain.{ext}',bbox_inches='tight',pad_inches=.06,metadata=meta)
    plt.close(fig)
    with Image.open(args.output_dir/'car_ca_cpu_complete_chain.png') as im:im.load();qa={'PNG_decode':'passed','size':im.size,'dpi':im.info.get('dpi')}
    manifest={'run_id':'car_ca_cpu_score_diagnostic_v1','source_csv_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest(),'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'panels':panels,'all_values_from_same_CPU_run':True,'archived_GPU_values_used_for_display':False,'QA':qa}
    (args.output_dir/'visual_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output_dir/'caption.md').write_text('One independent CPU CA diagnostic on the fixed ShapeNet-Car case at50% budget. PanelA summarizes the actual256-dimensional own-backbone block0 features by their L2 norm; full vectors remain in the indexed internal cache. The matching learned head outputs a standardized pressure proxy in panelB, which is decoded before the native gradient-score calculation. The same CPU score defines the target and actual native stratum/FPS selection. The final prediction and pressure error are recomputed for that selection; ground truth enters only the error diagnostic. Orange points in panelF are retained coarse points. All3,218 coarse points and16,093 total points are retained. These CPU scores differ from the archived GPU scores at154 native points under the unchanged tolerance; that comparison is separate and does not mix GPU values into this pipeline. The native ordered selected indices happen to agree exactly across these executions.\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
