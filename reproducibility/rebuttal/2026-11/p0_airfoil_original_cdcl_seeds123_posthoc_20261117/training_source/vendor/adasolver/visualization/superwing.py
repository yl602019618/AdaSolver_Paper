"""Evidence-bound SuperWing curves and three-channel case galleries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from adasolver.utils.interpolation import index_grid
from adasolver.inference.superwing_sampling import normalize_score

COLORS=dict(uniform='#666666',pi='#0072B2',cpg='#E69F00',ca='#009E73')
MARKERS=dict(uniform='o',pi='s',cpg='^',ca='D')
METHODS=('uniform','pi','cpg','ca')
LABELS=dict(uniform='Uniform',pi='PI',cpg='CPG',ca='CA')
CHANNELS=('Cp','150 Cf_tau','300 Cf_z')


def save(fig,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def plot_curves(summary,output):
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    x=np.asarray(summary['budgets'])*100
    for ax,target in zip(axes,('field','cd')):
        for method in METHODS:
            rows=[v for v in summary['policies'].values() if v['method']==method and v['target'] in (target,'shared')]
            if len(rows)>1:
                raise ValueError('curve figure requires one frozen policy per method and target')
            if not rows: continue
            metric=rows[0]['metrics'][target]
            y=np.asarray(metric['mean_curve'])*100
            std=np.asarray(metric['seed_std'])*100
            ax.plot(x,y,marker=MARKERS[method],color=COLORS[method],linewidth=1.8,
                    label=f'{LABELS[method]} (rAUEC {metric["rauec"]:.3f})')
            ax.fill_between(x,y-std,y+std,color=COLORS[method],alpha=.12)
        ax.set(xlabel='Final point budget (%)',ylabel='Full-field relative L2 (%)' if target=='field' else 'Cd relative error (%)',
               title='Field reconstruction' if target=='field' else 'Drag coefficient',xticks=x)
        ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    stage='Test' if summary['stage']=='frozen_test' else 'Calibration'
    fig.suptitle(f'SuperWing {stage} · {len(summary["sample_ids"])} samples · {len(summary["seeds"])} sampling seeds',fontsize=11)
    save(fig,Path(output)/'budget_error_curves')


def choose_representatives(fullmesh_records):
    ordered=sorted(fullmesh_records,key=lambda r:(r['field_relative_l2'],r['sample_id']))
    selected=[]
    used=set()
    for q in (.25,.5,.75):
        rank=round(q*(len(ordered)-1))
        candidates=sorted(range(len(ordered)),key=lambda i:(abs(i-rank),ordered[i]['sample_id']))
        row=next(ordered[i] for i in candidates if ordered[i]['shape_id'] not in used)
        selected.append(row['sample_id'])
        used.add(row['shape_id'])
    return selected


def plot_density_components(sample_id,entries,output):
    chosen=[e for e in entries if 'mixed_density' in e and e['record']['budget_fraction']==.5]
    if not chosen:return
    maximum=max(1.,float(np.quantile(np.concatenate([e[k] for e in chosen for k in
                ('geometry_density','prediction_density','mixed_density')]),.995)))
    fig,axes=plt.subplots(len(chosen),3,figsize=(12,2.5*len(chosen)),squeeze=False,layout='constrained')
    for row,e in enumerate(chosen):
        method=LABELS[e['record']['method']]
        for col,(key,name) in enumerate(zip(('geometry_density','prediction_density','mixed_density'),
                                           ('Geometry component','Prediction component','Combined proposal'))):
            ax=axes[row,col]
            artist=ax.imshow(e[key].reshape(128,256),origin='lower',cmap='viridis',vmin=0,vmax=maximum)
            title=f'{method} · {name}'
            if key=='mixed_density':title+=f' (λ={e["record"]["density_mix"]:g})'
            ax.set(xticks=[],yticks=[],title=title)
            fig.colorbar(artist,ax=ax,shrink=.8,extend='max')
    fig.suptitle(f'Sample {sample_id} · density components (mean 1) · 50% budget')
    save(fig,Path(output)/'density_components')


def plot_geometry(sample,fullmesh,output):
    g=sample.geometry
    gt=sample.target.reshape(128,256,3)
    full=np.asarray(fullmesh).reshape(128,256,3)
    fig=plt.figure(figsize=(12,6),layout='constrained')
    for row,values in enumerate((gt,full)):
        for c in range(3):
            ax=fig.add_subplot(2,3,row*3+c+1,projection='3d')
            limit=max(float(np.max(np.abs(gt[...,c]))),1e-12)
            norm=Normalize(-limit,limit)
            colors=plt.get_cmap('RdBu_r')(norm(values[...,c]))
            v=g.vertices
            ax.plot_surface(v[...,0],v[...,2],v[...,1],facecolors=colors,rstride=2,cstride=2,
                            shade=False,linewidth=0,edgecolor='none',antialiased=False)
            ax.set_proj_type('ortho')
            ax.view_init(elev=75,azim=-15)
            ax.set_box_aspect((np.ptp(v[...,0]),np.ptp(v[...,2]),max(np.ptp(v[...,1]),.1)))
            ax.set_title(f'{"GT" if row==0 else "Fullmesh"} · {CHANNELS[c]}',fontsize=10)
            ax.set_axis_off()
            fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='RdBu_r'),ax=ax,shrink=.55,pad=.01)
    fig.suptitle(f'Sample {g.sample_id} · shape {g.shape_id} · Mach {g.mach:.3f} · AoA {g.aoa:.2f}°',fontsize=12)
    save(fig,Path(output)/'geometry_gt_fullmesh')


def plot_case(sample,fullmesh,entries,output,targets=('field','cd')):
    """entries provide actual record/indices/reconstruction/score/compensation."""
    output=Path(output)
    gt=sample.target.reshape(128,256,3)
    full=np.asarray(fullmesh).reshape(128,256,3)
    error_limits=[]
    for channel in range(3):
        errors=[np.abs(full[...,channel]-gt[...,channel]).reshape(-1)]
        errors.extend(np.abs(e['prediction'].reshape(128,256,3)[...,channel]-gt[...,channel]).reshape(-1) for e in entries)
        error_limits.append(max(float(np.quantile(np.concatenate(errors),.995)),1e-12))
    plot_geometry(sample,fullmesh,output)
    for target in targets:
        target_label='Cd/Cl' if target=='cdcl' else target
        for fraction in (.375,.5,.75):
            chosen={m:next((e for e in entries if e['record']['method']==m and
                           (target=='cdcl' or e['record']['metric_target'] in (target,'shared')) and
                           e['record']['budget_fraction']==fraction),None) for m in METHODS}
            if any(v is None for v in chosen.values()):
                raise ValueError('case gallery requires every method at every requested budget/target')
            directory=output/target/f'budget_{fraction:g}'
            fig,axes=plt.subplots(4,4,figsize=(13,8),layout='constrained')
            comp_limit=max(float(np.max(e['compensation'])) for e in chosen.values())
            for col,method in enumerate(METHODS):
                e=chosen[method]
                selected=e['indices']
                y,x=np.divmod(selected,256)
                axes[0,col].scatter(x,y,s=.25,color=COLORS[method],rasterized=True)
                coarse=e.get('coarse',np.empty(0,int))
                if len(coarse):
                    cy,cx=np.divmod(coarse,256)
                    axes[0,col].scatter(cx,cy,s=.35,color='#333333',rasterized=True)
                axes[0,col].set_title(f'{LABELS[method]} · {len(selected):,} cells',fontsize=10)
                score=normalize_score(np.asarray(e['score']).reshape(-1)).reshape(128,256)
                axes[1,col].imshow(score,origin='lower',cmap='viridis',vmin=0,vmax=1,aspect='equal')
                axes[1,col].set_title('No acquisition' if method=='uniform' else 'Normalized acquisition score',fontsize=9)
                reference=index_grid((128,256))
                _,nearest=cKDTree(reference[selected]).query(reference)
                counts=np.bincount(nearest,minlength=len(selected))
                density=(32768/len(selected))/counts[nearest]
                density_artist=axes[2,col].imshow(density.reshape(128,256),origin='lower',cmap='viridis',
                                                  vmin=0,vmax=32768/len(selected))
                axes[2,col].set_title('Achieved relative cell density',fontsize=9)
                fig.colorbar(density_artist,ax=axes[2,col],shrink=.6)
                comp=np.full(32768,np.nan)
                comp[selected]=e['compensation']
                artist=axes[3,col].imshow(comp.reshape(128,256),origin='lower',cmap='viridis',vmin=0,vmax=comp_limit,aspect='equal')
                axes[3,col].set_title('Compensation at selected cells',fontsize=9)
                fig.colorbar(artist,ax=axes[3,col],shrink=.6)
                for row in range(4):
                    axes[row,col].set(xlim=(-.5,255.5),ylim=(-.5,127.5),xticks=[],yticks=[])
                    axes[row,col].set_aspect('equal')
            title=f'Sample {sample.sample_id} · {target_label} policy · {fraction:.1%} final budget'
            if target=='cdcl':title+='\nGray points in CPG/CA: retained 10% coarse points'
            fig.suptitle(title,fontsize=12)
            save(fig,directory/'sampling_score_compensation')
            fig=plt.figure(figsize=(13,3.8),layout='constrained')
            for col,method in enumerate(METHODS):
                ax=fig.add_subplot(1,4,col+1,projection='3d')
                points=sample.geometry.pos[chosen[method]['indices']]
                ax.scatter(points[:,0],points[:,2],points[:,1],s=.15,color=COLORS[method],depthshade=False,rasterized=True)
                ax.set_proj_type('ortho')
                ax.view_init(elev=75,azim=-15)
                v=sample.geometry.vertices
                ax.set_box_aspect((np.ptp(v[...,0]),np.ptp(v[...,2]),max(np.ptp(v[...,1]),.1)))
                ax.set_axis_off()
                ax.set_title(f'{LABELS[method]} · {len(points):,} cells',fontsize=10)
            fig.suptitle(f'Sample {sample.sample_id} · {target_label} · {fraction:.1%} sampled surface',fontsize=12)
            save(fig,directory/'sampling_surface')
            weights=sample.geometry.weights[1]
            gt_contributions=(weights*sample.target).sum(axis=0)
            full_contributions=(weights*np.asarray(fullmesh).reshape(-1,3)).sum(axis=0)
            names=['Fullmesh']+[LABELS[m] for m in METHODS]
            pressure=[full_contributions[0]]+[chosen[m]['record']['pressure_Cd'] for m in METHODS]
            friction=[full_contributions[1:].sum()]+[chosen[m]['record']['friction_Cd'] for m in METHODS]
            cd=np.asarray(pressure)+np.asarray(friction)
            fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
            positions=np.arange(len(names))
            axes[0].plot(positions,cd,'o-',color='#222222',label='Prediction')
            axes[0].axhline(sample.structured[1],color='#666666',linestyle='--',label='Structured GT')
            axes[0].set(xticks=positions,xticklabels=names,ylabel='Cd',title='Drag coefficient')
            axes[0].legend(frameon=False,fontsize=8)
            axes[1].bar(positions-.17,(np.asarray(pressure)-gt_contributions[0])*1e4,.34,color='#0072B2',label='Pressure')
            axes[1].bar(positions+.17,(np.asarray(friction)-gt_contributions[1:].sum())*1e4,.34,color='#E69F00',label='Friction')
            axes[1].axhline(0,color='#777777',linewidth=.6)
            axes[1].set(xticks=positions,xticklabels=names,ylabel='Signed error (drag counts)',title='Contributions vs GT-field closure')
            axes[1].legend(frameon=False,fontsize=8)
            fig.suptitle(f'Sample {sample.sample_id} · {target_label} · {fraction:.1%} budget',fontsize=12)
            save(fig,directory/'cd_comparison')
            if target=='cdcl':
                from adasolver.metrics.superwing import integrate_field
                from adasolver.metrics.superwing_cdcl import ratio_metrics
                full_coef=integrate_field(np.asarray(fullmesh).reshape(-1,3),sample.geometry.weights)
                coef=[full_coef]+[np.array([chosen[m]['record']['Cl_pred'],chosen[m]['record']['Cd_pred']]) for m in METHODS]
                fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
                values=[[c[1] for c in coef],[c[0] for c in coef],
                        [ratio_metrics(c[1],c[0],sample.structured[1],sample.structured[0])['CdCl_pred'] for c in coef]]
                truth=[sample.structured[1],sample.structured[0],sample.structured[1]/sample.structured[0]]
                for ax,label,prediction,reference in zip(axes,('Cd','Cl','Cd/Cl'),values,truth):
                    ax.plot(positions,prediction,'o-',color='#222222',label='Prediction')
                    ax.axhline(reference,color='#666666',linestyle='--',label='Structured GT')
                    ax.set(xticks=positions,xticklabels=names,ylabel=label)
                    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
                fig.suptitle(f'Sample {sample.sample_id} · {fraction:.1%} final budget',fontsize=12)
                save(fig,directory/'cdcl_comparison')
            for channel in range(3):
                fields=[gt[...,channel],full[...,channel]]+[
                    chosen[m]['prediction'].reshape(128,256,3)[...,channel] for m in METHODS]
                names=['GT','Fullmesh']+[LABELS[m] for m in METHODS]
                limit=max(float(np.max(np.abs(gt[...,channel]))),1e-12)
                # One pooled error scale per sample/channel across every budget
                # and objective. The colorbar marks the clipped 0.5% tail.
                err_limit=error_limits[channel]
                fig,axes=plt.subplots(2,6,figsize=(16,5),layout='constrained')
                for col,(name,field) in enumerate(zip(names,fields)):
                    field_artist=axes[0,col].imshow(field,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit)
                    error_artist=axes[1,col].imshow(np.abs(field-gt[...,channel]),origin='lower',cmap='magma',vmin=0,vmax=err_limit)
                    axes[0,col].set_title(name,fontsize=10)
                    for row in range(2): axes[row,col].set(xticks=[],yticks=[])
                axes[0,0].set_ylabel(CHANNELS[channel])
                axes[1,0].set_ylabel('Absolute error')
                fig.colorbar(field_artist,ax=axes[0,:],shrink=.7,extend='both')
                fig.colorbar(error_artist,ax=axes[1,:],shrink=.7,extend='max')
                fig.suptitle(f'Sample {sample.sample_id} · {target_label} · {fraction:.1%} budget · {CHANNELS[channel]}',fontsize=12)
                save(fig,directory/f'field_channel_{channel}')
