"""Evidence-bound DriverML plots from independent comparison summaries."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'uniform':'#666666','pi':'#E69F00','cpg':'#0072B2','ca':'#009E73'}
LABELS={'uniform':'Uniform','pi':'PI','cpg':'CPG','ca':'CA'}
MARKERS={'uniform':'o','pi':'^','cpg':'s','ca':'D'}
TARGETS={'pressure_field':'Pressure reconstruction','cd':'Drag coefficient (predicted WSS)'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,
                         'savefig.dpi':200,'figure.facecolor':'white'})


def save(figure,stem):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf','.svg'):
        figure.savefig(stem.with_suffix(suffix),bbox_inches='tight')
    plt.close(figure)


def combine_summaries(paths):
    merged=None
    for path in paths:
        data=json.loads(Path(path).read_text())
        if merged is None: merged=data
        else:
            for key in ('protocol_signature','sample_ids','seeds','budgets'):
                if data[key]!=merged[key]: raise ValueError('figure sources use different protocols or samples')
            for target,rows in data['targets'].items():
                if target in merged['targets'] and merged['targets'][target]!=rows:
                    raise ValueError('conflicting figure values for the same target')
                merged['targets'][target]=rows
    return merged


def plot_comparison(paths,output,*,stage,uncertainty=None):
    style();output=Path(output);summary=combine_summaries(paths)
    if stage=='official' and summary.get('stage')!='official':
        raise ValueError('official figures require an explicitly official summary')
    if stage=='official' and (len(summary['sample_ids'])!=50 or summary['seeds']!=[0,1,2]
                             or any(set(summary['targets'].get(t,{}))!=set(COLORS) for t in TARGETS)):
        raise ValueError('official figures require all four methods and both complete test targets')
    targets=[t for t in TARGETS if t in summary['targets']]
    intervals=json.loads(Path(uncertainty).read_text()) if uncertainty else None
    if intervals:
        for target in targets:
            for method,row in summary['targets'][target].items():
                if not np.isclose(intervals['targets'][target][method]['rauec'],row['rauec'],rtol=0,atol=1e-12):
                    raise ValueError('uncertainty values do not match the comparison')
    uncertainty_label='95% paired-car bootstrap intervals' if intervals else 'Error bars: sampling-seed SD'
    title=f"DrivAerML — {dict(official='Test',calibration='Development',confirmation='Confirmation')[stage]} ({len(summary['sample_ids'])} cars, {len(summary['seeds'])} seeds)"
    fig,axes=plt.subplots(1,len(targets),figsize=(7.1,3.1),squeeze=False,layout='constrained')
    for ax,target in zip(axes[0],targets):
        rows=summary['targets'][target];methods=[m for m in COLORS if m in rows]
        values=[rows[m]['rauec'] for m in methods]
        if intervals:
            lower=np.array([intervals['targets'][target][m]['lower'] for m in methods])
            upper=np.array([intervals['targets'][target][m]['upper'] for m in methods])
            errors=np.array([np.asarray(values)-lower,upper-np.asarray(values)])
        else:
            errors=np.array([rows[m].get('rauec_std_across_seeds',0.) for m in methods])
            upper=np.asarray(values)+errors
        bars=ax.bar(np.arange(len(methods)),values,width=.58,color=[COLORS[m] for m in methods],
                    yerr=errors,capsize=3,error_kw={'elinewidth':1,'ecolor':'#222222'})
        ax.axhline(1,color='#333333',lw=1,ls='--');ax.set_xticks(np.arange(len(methods)),[LABELS[m] for m in methods])
        for bar,value,high in zip(bars,values,upper):
            ax.text(bar.get_x()+bar.get_width()/2,high+.025,f'{value:.4f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(1.18,max(upper)+.15))
        ax.set_ylabel('rAUEC (lower is better)');ax.set_title(TARGETS[target]);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(title+'\n'+uncertainty_label,fontsize=11);save(fig,output/'rauec')
    fig,axes=plt.subplots(1,len(targets),figsize=(7.1,2.9),squeeze=False,layout='constrained')
    budgets=np.asarray(summary['budgets'])*100
    for ax,target in zip(axes[0],targets):
        rows=summary['targets'][target]
        for method in COLORS:
            if method not in rows: continue
            y=np.asarray(rows[method]['mean_error'])*100
            sd=np.asarray(rows[method]['error_std_across_seeds'])*100
            ax.plot(budgets,y,color=COLORS[method],marker=MARKERS[method],ls='--' if method=='uniform' else '-',lw=1.8,label=LABELS[method])
            if len(summary['seeds'])>1: ax.fill_between(budgets,y-sd,y+sd,color=COLORS[method],alpha=.12,lw=0)
        ax.set_xticks(budgets,['37.5','50','75']);ax.set_xlabel('Final points (% of full surface)')
        ax.set_ylabel('Relative L2 (%)' if target=='pressure_field' else 'Relative Cd error (%)')
        ax.set_title(TARGETS[target]);ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.suptitle(title+'\nBands: sampling-seed SD',fontsize=11);save(fig,output/'budget_curves')
    output.mkdir(parents=True,exist_ok=True)
    manifest=dict(stage=stage,sample_ids=summary['sample_ids'],seeds=summary['seeds'],sources=[str(Path(p)) for p in paths],
                  uncertainty=dict(rauec=uncertainty_label,curves='sampling-seed SD',source=str(uncertainty) if uncertainty else None),
                  units=dict(rauec='dimensionless',curves='percent relative error'),methods=LABELS,colors=COLORS)
    (output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def plot_runtime(directory,output):
    from adasolver.evaluation.driverml_compare import read_records
    style();directory=Path(directory);output=Path(output)
    summary=json.loads((directory/'comparison.json').read_text())
    result={}
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    segment_colors=['#CC79A7','#56B4E9','#009E73','#CCCCCC']
    for col,target in enumerate(TARGETS):
        result[target]={};segments=[];memory=[]
        for method in COLORS:
            rows=read_records(sorted((directory/target/method).glob('shard_*.jsonl')))
            if len(rows)!=summary['targets'][target][method]['record_count']:
                raise ValueError('runtime records do not match the comparison')
            mean=lambda key:float(np.mean([r[key] for r in rows]))
            acquisition=mean('acquisition_seconds')+mean('geometry_seconds')
            final=mean('pressure_inference_seconds')+mean('wss_inference_seconds')
            reconstruction=mean('density_seconds')+mean('reconstruction_seconds')
            total=mean('total_seconds');other=total-acquisition-final-reconstruction
            if other < -1e-6: raise ValueError('runtime components exceed total')
            peak=max(r['peak_gpu_bytes'] for r in rows)/2**30
            result[target][method]=dict(acquisition_seconds=acquisition,final_seconds=final,
                reconstruction_density_seconds=reconstruction,other_seconds=max(other,0),
                total_seconds=total,io_seconds=mean('io_seconds'),peak_gpu_gib=peak)
            segments.append([acquisition,final,reconstruction,max(other,0)]);memory.append(peak)
        x=np.arange(4);bottom=np.zeros(4)
        for j,label in enumerate(['Acquisition / geometry','Final operators','Density / reconstruction','Other computation']):
            values=np.asarray(segments)[:,j]
            axes[0,col].bar(x,values,bottom=bottom,color=segment_colors[j],label=label)
            bottom+=values
        for i,value in enumerate(bottom): axes[0,col].text(i,value+.25,f'{value:.2f}',ha='center',fontsize=9)
        axes[0,col].set_ylim(0,max(bottom)*1.15);axes[0,col].set_ylabel('Mean compute time (s)')
        axes[0,col].set_title(TARGETS[target]);axes[0,col].set_xticks(x,list(LABELS.values()))
        axes[1,col].bar(x,memory,color=list(COLORS.values()))
        for i,value in enumerate(memory): axes[1,col].text(i,value+.3,f'{value:.1f}',ha='center',fontsize=9)
        axes[1,col].set_ylim(0,max(memory)*1.15);axes[1,col].set_ylabel('Peak allocated GPU memory (GiB)')
        axes[1,col].set_xticks(x,list(LABELS.values()))
    for ax in axes.ravel(): ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=8,frameon=False)
    fig.suptitle('DrivAerML test: parallel evaluation compute cost\nI/O excluded from time; 50 cars, 3 seeds, 3 budgets',fontsize=11)
    save(fig,output/'runtime_memory')
    (output/'runtime_summary.json').write_text(json.dumps(dict(source=str(directory),targets=result,
        note='Parallel evaluation, not isolated latency; memory is max across records.'),indent=2)+'\n')


def _project(xyz):
    eye=np.array([.35,-1.,.75]);eye/=np.linalg.norm(eye)
    right=np.cross([0.,0.,1.],eye);right/=np.linalg.norm(right)
    up=np.cross(eye,right)
    return np.column_stack((xyz@right,xyz@up)),xyz@eye


def _cloud(ax,xyz,values=None,*,color=None,cmap='viridis',vmin=None,vmax=None,size=.5):
    xy,depth=_project(np.asarray(xyz))
    order=np.argsort(depth)
    args=dict(s=size,linewidths=0,rasterized=True)
    if values is None: args['color']=color or '#999999'
    else: args.update(c=np.asarray(values)[order],cmap=cmap,vmin=vmin,vmax=vmax)
    artist=ax.scatter(xy[order,0],xy[order,1],**args)
    ax.set_aspect('equal');ax.set_axis_off()
    return artist


def plot_surface_gallery(snapshot_directory,run_ids,output):
    style();source=Path(snapshot_directory);output=Path(output)
    methods=list(COLORS);budgets=[.375,.5,.75]
    for run_id in run_ids:
        directory=output/f'sample_{run_id}';directory.mkdir(parents=True,exist_ok=True)
        with np.load(source/f'fullmesh_run{run_id}.npz') as data:
            full={key:data[key] for key in data.files}
        full_meta=json.loads((source/f'fullmesh_run{run_id}.json').read_text())
        pressure_limit=float(np.quantile(np.abs(full['pressure_gt']),.995))
        fig,axes=plt.subplots(1,3,figsize=(9,2.8),layout='constrained')
        geometry_artist=_cloud(axes[0],full['xyz'],full['normal_x'],cmap='RdBu_r',vmin=-1,vmax=1)
        axes[0].set_title('Input geometry')
        fig.colorbar(geometry_artist,ax=axes[0],label='Stored normal x',shrink=.7,ticks=[-1,0,1])
        for ax,key,title in [(axes[1],'pressure_gt','GT pressure'),(axes[2],'pressure_pred','Full-mesh pressure')]:
            artist=_cloud(ax,full['xyz'],full[key],cmap='RdBu_r',vmin=-pressure_limit,vmax=pressure_limit)
            ax.set_title(title)
        fig.colorbar(artist,ax=axes[1:],label='Pressure (Pa)',shrink=.7,extend='both')
        fig.suptitle(f"DrivAerML run {run_id} — {full_meta['full_count']:,} reference points\nFixed display subset: {len(full['xyz']):,} points; full-reference metrics",fontsize=11)
        save(fig,directory/'input_gt_fullmesh')
        for target in TARGETS:
            entries={}
            for b in budgets:
                for method in methods:
                    stem=f'{target}_{method}_run{run_id}_s0_b{b:.3f}'
                    with np.load(source/f'{stem}.npz') as archive:
                        data={key:archive[key] for key in archive.files}
                    meta=json.loads((source/f'{stem}.json').read_text())
                    entries[b,method]=(data,meta)
            errors=np.concatenate([np.abs(d['pressure_pred']-d['pressure_gt']) for d,m in entries.values()])
            error_limit=float(np.quantile(errors,.995))
            for view in ['sampling','density','acquisition','prediction','error']:
                fig,axes=plt.subplots(3,4,figsize=(10.3,6.8),layout='constrained')
                artist=None
                for row,b in enumerate(budgets):
                    for col,method in enumerate(methods):
                        ax=axes[row,col];data,meta=entries[b,method]
                        if view=='sampling':
                            _cloud(ax,data['xyz'],color='#dddddd',size=.35)
                            # Thin the already-uniform display subset proportionally to
                            # the budget, so displayed counts do not mask budget changes.
                            shown=round(len(data['selected_xyz'])*b/max(budgets))
                            xyz=data['selected_xyz'][:shown];coarse=data['is_coarse'][:shown]
                            _cloud(ax,xyz[~coarse],color=COLORS[method],size=1.1)
                            if coarse.any(): _cloud(ax,xyz[coarse],color='#333333',size=.8)
                        else:
                            if view=='density': vals=data['relative_density'];cmap='viridis';lo,hi=0,3
                            elif view=='acquisition': vals=data['acquisition_score'];cmap='viridis';lo,hi=0,1
                            elif view=='prediction': vals=data['pressure_pred'];cmap='RdBu_r';lo,hi=-pressure_limit,pressure_limit
                            else: vals=np.abs(data['pressure_pred']-data['pressure_gt']);cmap='magma';lo,hi=0,error_limit
                            artist=_cloud(ax,data['xyz'],vals,cmap=cmap,vmin=lo,vmax=hi,size=.7)
                        error=meta['pressure_field_relative_l2'] if target=='pressure_field' else meta['cd_relative_error']
                        metric_name='L2' if target=='pressure_field' else 'Cd error'
                        ax.set_title(f"{LABELS[method]} · {b*100:g}%\nN={meta['selected_count']:,}, {metric_name}={100*error:.2f}%",fontsize=8.5)
                        if view=='sampling': ax.text(.5,-.04,f'{shown:,} selected points displayed',transform=ax.transAxes,ha='center',fontsize=7.5)
                labels={'density':'Point density / uniform','acquisition':'Normalized raw acquisition score',
                        'prediction':'Pressure (Pa)','error':'Absolute pressure error (Pa)'}
                if artist is not None: fig.colorbar(artist,ax=axes.ravel().tolist(),label=labels[view],shrink=.8,pad=.01,extend='max' if view in ('density','error') else ('both' if view=='prediction' else 'neither'))
                title=f"Run {run_id} · {TARGETS[target]} · {view.capitalize()}"
                if view=='sampling': title+='\nDisplay subsets; N = actual final points; coarse points in charcoal'
                else: title+=f"\nFixed {len(full['xyz']):,}-point display subset; metrics use full reference"
                if view=='acquisition':
                    head_meta=entries[budgets[0],'ca'][1]
                    kind='single score' if head_meta.get('head_target')=='normalized_log1p_cpg_lap_proxy' else 'pressure'
                    title+=f"\n{LABELS['ca']}: {kind} head, d={head_meta['head_feature_dim']}, {head_meta.get('head_feature_depth',1)} frozen block(s); density power={head_meta.get('score_power',1):g}"
                fig.suptitle(title,fontsize=11)
                save(fig,directory/f'{target}_{view}')
            if target=='cd':
                fig,ax=plt.subplots(figsize=(7.1,3),layout='constrained')
                width=.18
                for j,method in enumerate(methods):
                    x=np.arange(3)+(j-1.5)*width
                    p=np.array([entries[b,method][1]['cd_pressure'] for b in budgets])
                    w=np.array([entries[b,method][1]['cd_viscous'] for b in budgets])
                    ax.bar(x,p,width,color=COLORS[method],label=LABELS[method])
                    ax.bar(x,w,width,bottom=p,color=COLORS[method],hatch='///',edgecolor='#333333',linewidth=.3,alpha=.55)
                ax.axhline(entries[budgets[0],'uniform'][1]['cd_gt'],color='#222222',ls='--',lw=1,label='GT total')
                ax.set_xticks(range(3),['37.5%','50%','75%']);ax.set_ylabel('Cd');ax.set_xlabel('Final point budget')
                ax.set_title(f'Run {run_id}: predicted pressure + viscous drag (hatched)')
                ax.legend(frameon=False,ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.25));save(fig,directory/'cd_contributions')
        (directory/'visualization_manifest.json').write_text(json.dumps(dict(run_id=run_id,seed=0,budgets=budgets,
            display_points=len(full['xyz']),reference_points=full_meta['full_count'],camera=[.35,-1.,.75],
            clipping='shared 99.5th-percentile display ranges for pressure/error; metrics use all points',
            sampling_display='uniform subset of saved selected points, thinned in proportion to final budget; N labels actual final counts',
            source=str(source)),indent=2)+'\n')
        gallery=[f'# DrivAerML run {run_id}', '',
                 '固定显示子集和相机；标题 N 为实际推理点数，指标在完整参考面计算。所有图均另有同名 PDF / SVG。', '',
                 '![Input, GT and fullmesh pressure](input_gt_fullmesh.png)', '']
        for target in TARGETS:
            for view in ['sampling','acquisition','density','prediction','error']:
                gallery.extend([f'![{TARGETS[target]}: {view}]({target}_{view}.png)', ''])
        gallery.extend(['![Pressure and viscous drag contributions](cd_contributions.png)', ''])
        (directory/'README.md').write_text('\n'.join(gallery))


def plot_head_selection(path,output):
    style();data=json.loads(Path(path).read_text())
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.9),layout='constrained')
    for ax,target in zip(axes,TARGETS):
        record=data[target];rows=sorted(record['options'],key=lambda r:r['config']['head_feature_dim'])
        dimensions=[r['config']['head_feature_dim'] for r in rows];values=[r['rauec'] for r in rows]
        chosen=record['selected'];minimum=min(values)
        ax.axhspan(minimum,minimum+record['tolerance'],color=COLORS['ca'],alpha=.10,label='Within 0.005 of best')
        ax.plot(dimensions,values,color=COLORS['ca'],marker='D',lw=1.8)
        ax.scatter([chosen['config']['head_feature_dim']],[chosen['rauec']],s=140,facecolors='none',edgecolors='#222222',lw=1.2,zorder=5)
        ax.set_xscale('log',base=2);ax.set_xticks(dimensions,[str(d) for d in dimensions])
        ax.set_xlabel('MLP bottleneck dimension');ax.set_ylabel('Development rAUEC')
        ax.set_title(TARGETS[target]);ax.grid(alpha=.16)
        ax.text(.98,.96,f"Selected d={chosen['config']['head_feature_dim']}",transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.legend(loc='upper right',bbox_to_anchor=(1,.88),frameon=False,fontsize=8)
    fig.suptitle('CA feature selection — 18 development cars, 3 seeds',fontsize=12)
    save(fig,Path(output)/'head_dimensions')
