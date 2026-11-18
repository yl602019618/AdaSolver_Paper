"""Deterministic paper figures from accepted CSVs and exported sampling maps."""
from __future__ import annotations
import argparse,json,os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_key]='4'
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from common import OUT,CODE,read,write,digest

COLORS={'uniform':'#666666','pi':'#E69F00','cpg':'#0072B2','ca':'#009E73','full':'#222222'}
LABELS={'uniform':'Uniform','pi':'PI','cpg':'CPG','ca':'CA','full':'Full'}
MARKERS={'uniform':'o','pi':'s','cpg':'^','ca':'D'}
LINESTYLES={'uniform':'--','pi':'-.','cpg':'-','ca':':'}
METHODS=('uniform','pi','cpg','ca')
RESOLUTIONS=(85,141,211)
FIGURES=OUT/'figures'
SOURCES=FIGURES/'source_data'


def setup():
    FIGURES.mkdir(parents=True,exist_ok=True);SOURCES.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'darcy_resolution_matched_20261117','savefig.facecolor':'white',
        'lines.linewidth':1.6,'lines.markersize':5})


def save(fig,name,caption,sources):
    fig.canvas.draw()
    files={}
    for extension in ['pdf','svg','png']:
        path=FIGURES/f'{name}.{extension}'
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None} if extension=='svg' else None
        fig.savefig(path,dpi=300,bbox_inches='tight',pad_inches=.07,metadata=metadata)
        files[extension]=dict(path=str(path.relative_to(OUT)),sha256=digest(path))
    plt.close(fig)
    return dict(name=name,caption=caption,files=files,source_files={str(path.relative_to(OUT)):digest(path) for path in sources})


def legend(fig,full=False):
    methods=list(METHODS)+(['full'] if full else [])
    handles=[Line2D([0],[0],color=COLORS[m],linestyle='--' if m=='full' else LINESTYLES[m],
        marker=None if m=='full' else MARKERS[m],label=LABELS[m]) for m in methods]
    fig.legend(handles=handles,loc='upper center',ncol=len(methods),frameon=False,bbox_to_anchor=(.5,1.07))


def rauec_plot():
    path=OUT/'results/rauec_summary.csv';data=pd.read_csv(path)
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.3),sharey=True,layout='constrained')
    for ax,grid,title in zip(axes,['common','native'],['(a) Common 211² output','(b) Native output']):
        for method in METHODS:
            group=data[(data.evaluation_grid==grid)&(data.method==method)].sort_values('resolution')
            ax.errorbar(group.resolution,group.rauec_mean,yerr=group.sampling_sd,label=LABELS[method],
                color=COLORS[method],marker=MARKERS[method],linestyle=LINESTYLES[method],capsize=3)
        ax.axhline(1.,color='#777777',lw=.8,ls='--',zorder=0)
        ax.set_xticks(RESOLUTIONS,[f'{n}²' for n in RESOLUTIONS]);ax.set_xlabel('Training / observation resolution')
        ax.ticklabel_format(axis='y',useOffset=False,style='plain')
        ax.set_title(title);ax.grid(axis='y',color='#E6E6E6',linewidth=.6)
    axes[0].set_ylabel('rAUEC (lower is better)');legend(fig)
    return save(fig,'A_rauec_resolution',
        'Controlled resolution trend for three models trained from scratch on the same physical split. Left: common 211² output; right: native output. Each model/grid uses its own Uniform area. Error bars are sample SD across three sampling seeds for training seed 0. The historical paper 85² is excluded.',[path])


def error_plot():
    path=OUT/'results/absolute_errors.csv';data=pd.read_csv(path)
    fig,axes=plt.subplots(2,3,figsize=(11.1,6.2),sharex=True,layout='constrained')
    for i,grid in enumerate(['common','native']):
        subset=data[data.evaluation_grid==grid]
        lower=max(0.,float((subset.error_mean-subset.sampling_sd.fillna(0)).min())*.95)
        upper=float((subset.error_mean+subset.sampling_sd.fillna(0)).max())*1.06
        for j,n in enumerate(RESOLUTIONS):
            ax=axes[i,j];group=subset[subset.resolution==n]
            for method in METHODS:
                line=group[group.method==method].sort_values('budget_fraction')
                ax.errorbar(line.budget_fraction*100,line.error_mean,yerr=line.sampling_sd,
                    color=COLORS[method],marker=MARKERS[method],linestyle=LINESTYLES[method],capsize=3)
            full=float(group[group.method=='full'].error_mean.iloc[0])
            ax.axhline(full,color=COLORS['full'],ls='--',lw=1.1)
            ax.set_title(f'{n}² training; '+('common 211²' if grid=='common' else f'native {n}²'))
            ax.set_ylim(lower,upper);ax.set_xticks([37.5,50,75]);ax.grid(axis='y',color='#E6E6E6',linewidth=.6)
            if j==0:ax.set_ylabel('Physical field relative L2')
            if i==1:ax.set_xlabel('Retained observations (%)')
    legend(fig,True)
    return save(fig,'B_absolute_error_budget',
        'Absolute full-field relative L2 on common (top) and native (bottom) output grids. Each row uses a common y axis across resolutions. The dashed black line is each checkpoint Full error; Full is evaluated once per case. Error bars show sampling SD across three seeds after averaging the same 200 cases.',[path])


def region_plot():
    path=OUT/'results/regional_summary.csv';data=pd.read_csv(path)
    fig,axes=plt.subplots(3,3,figsize=(11.1,8.1),sharex=True,sharey=True,layout='constrained')
    for i,n in enumerate(RESOLUTIONS):
        for j,region in enumerate(['boundary','interface','smooth']):
            ax=axes[i,j];group=data[(data.resolution==n)&(data.region==region)]
            for method in METHODS:
                line=group[group.method==method].sort_values('budget_fraction')
                ax.errorbar(line.budget_fraction*100,line.mean_case_retention_enrichment,
                    yerr=line.mean_case_retention_enrichment_sampling_sd,color=COLORS[method],
                    marker=MARKERS[method],linestyle=LINESTYLES[method],capsize=3)
            ax.axhline(1.,color='#777777',ls='--',lw=.8)
            ax.set_title(f'{n}² — {region}');ax.set_xticks([37.5,50,75]);ax.grid(axis='y',color='#E6E6E6',linewidth=.6)
            if j==0:ax.set_ylabel('Regional retention / global retention')
            if i==2:ax.set_xlabel('Retained observations (%)')
    legend(fig)
    return save(fig,'C_regional_retention',
        'Actual regional retained fraction divided by the actual global retained fraction. Values above 1 indicate relative refinement; below 1 indicate lower sampling. Curves average physical-case regional rates before aggregating three seeds; error bars are sampling SD. Boundary and interface distances are evaluated directly at native coordinates with width 0.02 and the same common211 coefficient-edge geometry. This corrects nearest-boolean region classification without changing any retained point. These curves measure retained indices, not target densities.',[path])


def export_examples():
    """Make explicit CSV plotting inputs from the saved real selected-point arrays."""
    points=[];bins=[];source_rows=[]
    strict_labels={n:pd.read_csv(OUT/f'results/physical_region_labels_r{n}.csv',float_precision='round_trip') for n in RESOLUTIONS}
    strict_receipts={n:read(OUT/f'results/physical_region_r{n}_acceptance.json') for n in RESOLUTIONS}
    for n in RESOLUTIONS:
        assert strict_receipts[n]['passed']
        assert digest(OUT/f'results/physical_region_labels_r{n}.csv')==strict_receipts[n]['example_labels_csv_sha256']
    for case_id in [1000,1099,1199]:
        for n in RESOLUTIONS:
            label_rows=strict_labels[n][strict_labels[n].case_id==case_id].sort_values('node_index')
            assert len(label_rows)==n*n and np.array_equal(label_rows.node_index.to_numpy(),np.arange(n*n))
            source=OUT/f'records/r{n}/all/case{case_id}.json'
            if not source.exists():
                raw=[]
                for phase in ['baseline','ca']:raw+=read(OUT/f'records/r{n}/{phase}/case{case_id}.json')['records']
            else:raw=read(source)['records']
            for method in ['pi','cpg','ca']:
                row=next(r for r in raw if r['method']==method and r['budget_fraction']==.5 and r['sampling_seed']==0)
                path=OUT/row['example_array'];assert digest(path)==row['example_array_sha256']
                with np.load(path,allow_pickle=False) as a:
                    coords=a['reference_coords']
                    np.testing.assert_array_equal(label_rows[['x','y']].to_numpy(),coords)
                    raw_score=np.full(n*n,np.nan);raw_score[a['score_support_indices']]=a['score_support_values']
                    points.append(pd.DataFrame(dict(case_id=case_id,resolution=n,method=method,budget_fraction=.5,sampling_seed=0,
                        node_index=np.arange(n*n),x=coords[:,0],y=coords[:,1],region=label_rows.region.to_numpy(),
                        legacy_region=label_rows.legacy_region.to_numpy(),region_protocol_id=strict_receipts[n]['region_protocol_id'],raw_support_score=raw_score,
                        relative_display_score=a['relative_score'],target_relative_density=a['target_relative_density'],retained=a['retained_indicator'])))
                    for iy in range(16):
                        for ix in range(16):
                            bins.append(dict(case_id=case_id,resolution=n,method=method,budget_fraction=.5,sampling_seed=0,
                                bin_x=ix,bin_y=iy,observed_count=int(a['bin_observed_count'][iy,ix]),
                                selected_count=int(a['bin_selected_count'][iy,ix]),actual_retained_fraction=float(a['bin_retained_fraction'][iy,ix])))
                source_rows.append(dict(case_id=case_id,resolution=n,method=method,budget_fraction=.5,sampling_seed=0,
                    array_path=str(path.relative_to(OUT)),array_sha256=digest(path),checkpoint_sha256=row['checkpoint_sha256'],
                    selected_indices_sha256=row['selected_indices_sha256'],target_density_semantics=row['target_density_semantics'],
                    region_protocol_id=strict_receipts[n]['region_protocol_id'],
                    physical_labels_csv=strict_receipts[n]['example_labels_csv'],physical_labels_csv_sha256=strict_receipts[n]['example_labels_csv_sha256'],
                    score_display_normalization='nearest-lifted raw support score divided by mean raw support score'))
    pointpath=SOURCES/'qualitative_values.csv.gz';binpath=SOURCES/'qualitative_bins.csv';sourcepath=SOURCES/'qualitative_sources.csv'
    pd.concat(points,ignore_index=True).to_csv(pointpath,index=False,compression={'method':'gzip','mtime':0},float_format='%.17g')
    pd.DataFrame(bins).to_csv(binpath,index=False,float_format='%.17g');pd.DataFrame(source_rows).to_csv(sourcepath,index=False)
    return pointpath,binpath,sourcepath


def example_plots(case_ids=(1000,1099,1199)):
    pointpath,binpath,sourcepath=export_examples()
    points=pd.read_csv(pointpath);bins=pd.read_csv(binpath);sources=pd.read_csv(sourcepath);output=[];limits=[]
    region_cmap=ListedColormap(['#E7E7E7','#56B4E9','#E69F00'])
    for case_id in case_ids:
        case=points[points.case_id==case_id]
        score_max=max(1.,float(case.relative_display_score.max()))
        density_max=max(1.,float(case.target_relative_density.max()))
        limits.append(dict(case_id=case_id,score_min=0.,score_max=score_max,score_display_transform='log1p',
            transformed_score_min=0.,transformed_score_max=float(np.log1p(score_max)),density_min=0.,density_max=density_max,retained_min=0.,retained_max=1.,
            sharing='all three resolutions and PI/CPG/CA; no quantile clipping'))
        for method in ['pi','cpg','ca']:
            fig,axes=plt.subplots(3,5,figsize=(13.8,7.8),layout='constrained')
            score_image=density_image=retention_image=None
            for i,n in enumerate(RESOLUTIONS):
                values=case[(case.resolution==n)&(case.method==method)].sort_values('node_index')
                actual=bins[(bins.case_id==case_id)&(bins.resolution==n)&(bins.method==method)].sort_values(['bin_y','bin_x'])
                assert len(values)==n*n and len(actual)==256
                kwargs=dict(origin='lower',extent=(0,1,0,1),interpolation='nearest')
                axes[i,0].imshow(values.region.to_numpy().reshape(n,n),cmap=region_cmap,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),**kwargs)
                score_image=axes[i,1].imshow(np.log1p(values.relative_display_score.to_numpy()).reshape(n,n),cmap='viridis',vmin=0,vmax=np.log1p(score_max),**kwargs)
                density_image=axes[i,2].imshow(values.target_relative_density.to_numpy().reshape(n,n),cmap='viridis',vmin=0,vmax=density_max,**kwargs)
                retention_image=axes[i,3].imshow(actual.actual_retained_fraction.to_numpy().reshape(16,16),cmap='viridis',vmin=0,vmax=1,**kwargs)
                chosen=values[values.retained==1]
                axes[i,4].scatter(chosen.x,chosen.y,s=.16,color='#222222',linewidths=0)
                axes[i,4].set_xlim(0,1);axes[i,4].set_ylim(0,1);axes[i,4].set_aspect('equal')
                axes[i,0].set_ylabel(f'{n}² observations\ny')
                for ax in axes[i]:
                    ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
                    if i==2:ax.set_xlabel('x')
                if i==0:
                    for ax,title in zip(axes[i],['Physical regions','Score\n(log display)','Target density\n(relative)','Actual retained fraction\n(16 × 16 bins)','Retained nodes\n(actual)']):ax.set_title(title)
            fig.colorbar(score_image,ax=axes[:,1],shrink=.6,label='log(1 + score / support mean)',pad=.02)
            fig.colorbar(density_image,ax=axes[:,2],shrink=.6,label='Density / uniform density',pad=.02)
            fig.colorbar(retention_image,ax=axes[:,3],shrink=.6,label='Fraction',pad=.02)
            handles=[Line2D([0],[0],marker='s',linestyle='',color=color,label=label) for color,label in zip(region_cmap.colors,['Smooth','Interface','Boundary'])]
            fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.004,1.026),ncol=3,frameon=False)
            fig.suptitle(f'{LABELS[method]} — physical case {case_id}; 50% budget, sampling seed 0',x=.65,y=1.015,fontsize=11)
            cap=f'Fixed case {case_id}, {LABELS[method]}, 50% budget and sampling seed 0, with each resolution own trained model/head. Physical regions use direct native-coordinate distances and the shared common211 coefficient-edge geometry, with width 0.02. The original nearest-boolean labels are retained only as legacy CSV columns. Score is a nearest-lifted display normalized by its coarse/native support mean and shown as log(1 + relative score); raw support scores and untransformed relative scores are in the CSV. This display transform does not change sampling. Target density is '+('the PI nominal proposal, not an exact inclusion probability.' if method=='pi' else 'the pre-FPS final patch allocation divided by native available nodes and global budget.')+' Actual bin retention and actual retained nodes are plotted separately. Color limits are shared across all resolutions and methods for this case without quantile clipping.'
            output.append(save(fig,f'D_sampling_case{case_id}_{method}',cap,[pointpath,binpath,sourcepath]))
    pd.DataFrame(limits).to_csv(SOURCES/'qualitative_color_limits.csv',index=False,float_format='%.17g')
    return output


def training_plot():
    rows=[]
    for n in RESOLUTIONS:
        for kind,folder in [('backbone',OUT/f'checkpoints/r{n}'),('ca_head',OUT/f'checkpoints/r{n}/head')]:
            path=folder/'history.jsonl'
            if not path.exists():continue
            for line in path.read_text().splitlines():
                if line.strip():rows.append(dict(resolution=n,kind=kind,**json.loads(line)))
    if not rows:return None
    data=pd.DataFrame(rows);path=SOURCES/'training_head_curves.csv';data.to_csv(path,index=False)
    fig,axes=plt.subplots(3,2,figsize=(10.1,8),layout='constrained')
    for i,n in enumerate(RESOLUTIONS):
        backbone=data[(data.resolution==n)&(data.kind=='backbone')].sort_values('epoch')
        head=data[(data.resolution==n)&(data.kind=='ca_head')].sort_values('epoch')
        assert len(backbone)==len(head)==1000
        ax=axes[i,0];ax.plot(backbone.epoch,backbone.train_field_l2,color='#0072B2',label='Train field')
        ax.plot(backbone.epoch,backbone.validation_field_l2,color='#E69F00',label='Validation field')
        ax.set_title(f'{n}² backbone');ax.set_ylabel('Physical field relative L2');ax.legend(frameon=False)
        ax=axes[i,1];left=ax.plot(head.epoch,head.validation,color='#0072B2',label='Validation vector norm')
        right=ax.twinx();right.spines['right'].set_visible(True)
        rightlines=right.plot(head.epoch,head.train_loss,color='#E69F00',ls='--',label='Train Huber loss')
        ax.set_ylabel('Validation vector norm');right.set_ylabel('Train Huber loss');ax.set_title(f'{n}² CA head')
        ax.legend(left+rightlines,[line.get_label() for line in left+rightlines],frameon=False,loc='upper right')
        for axis in axes[i]:axis.set_xlabel('Epoch');axis.grid(axis='y',color='#E6E6E6',linewidth=.6)
    return save(fig,'E_training_head_curves',
        'Full 1000-epoch training curves. Backbone panels show physical field train and validation relative L2; head panels use separate labelled axes for validation vector norm and training Huber loss. Checkpoint selection uses validation only. Exact original epoch fields are retained in the source CSV.',[path])


def main():
    setup();acceptance=read(OUT/'results/statistics_acceptance.json')
    assert acceptance['passed'] and acceptance['complete']
    aggregation=read(OUT/'results/aggregation_receipt.json')
    assert aggregation['complete'] and digest(OUT/'results/aggregation_receipt.json')==acceptance['aggregation_receipt_sha256']
    for filename in ['rauec_summary.csv','absolute_errors.csv','regional_summary.csv']:
        assert digest(OUT/'results'/filename)==aggregation['result_files']['results/'+filename]
    figures=[rauec_plot(),error_plot(),region_plot()]
    figures+=example_plots()
    training=training_plot()
    if training:figures.append(training)
    lines=['# Figure index','',
        'All scientific result plots are generated from accepted CSV data. PDF and SVG retain vector text; PNG exports use 300 dpi. Color identifies methods and marker/line style provides a redundant encoding.','']
    for row in figures:
        lines+=['## '+row['name'],'',row['caption'],'']
        if row['files']:lines+=[' · '.join(f'[{fmt.upper()}]({Path(item["path"]).name})' for fmt,item in row['files'].items()),'']
    (FIGURES/'README.md').write_text('\n'.join(lines))
    rendered=sum(bool(row['files']) for row in figures)
    assert rendered==13
    result=dict(passed=True,complete=True,rendered_figure_groups=rendered,exported_result_files=rendered*3,
        formats=['pdf','svg','png'],png_dpi=300,figures=figures,plot_source_sha256=digest(__file__),
        statistics_acceptance_sha256=digest(OUT/'results/statistics_acceptance.json'),
        render_visual_review='pending inspection of actual generated PNGs; numeric/source acceptance has passed')
    write(OUT/'results/figures_receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='figures'},indent=2))


if __name__=='__main__':main()
