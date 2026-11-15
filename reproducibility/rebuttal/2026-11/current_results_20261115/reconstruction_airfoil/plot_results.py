#!/usr/bin/env python3
"""Current Car/Airfoil figures from verified summary CSVs; seed0 throughout."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

COLORS={'uniform':'#666666','pi':'#0072B2','cpg':'#D55E00','ca':'#009E73','full':'#222222'}
MARKERS={'uniform':'o','pi':'s','cpg':'^','ca':'D','full':'*'}
LABELS={'uniform':'Uniform','pi':'PI','cpg':'CPG beta1','ca':'CA','full':'Full'}
MODELS=[('shapenet_car','Transolver3','Car · Transolver3'),('shapenet_car','LinearNO','Car · LinearNO'),('airfoil2d','Transolver3','Airfoil · Transolver3'),('airfoil2d','LinearNO_point_temp','Airfoil · LinearNO point-temp')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();d=args.data_dir;out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    def read(name):
        with (d/name).open() as f:return list(csv.DictReader(f))
    def canon(m):return 'cpg' if m=='cpg_beta1' else m
    def primary(r):
        return r['metric'] in ['field_error','field_relative_l2'] and r['method'] not in ['cpg_beta0'] and not (r['method']=='cpg' and r.get('beta')=='0.0')
    def select(rows,dataset,backbone):return [r for r in rows if r['dataset']==dataset and r['backbone']==backbone]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'current-reconstruction-20261115','savefig.dpi':300})
    qa=[];captions=[]
    def export(fig,name,rows,caption):
        for ext in ['pdf','svg','png']:
            meta={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {}
            fig.savefig(out/(name+'.'+ext),bbox_inches='tight',pad_inches=.06,metadata=meta)
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (out/(name+'_source.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
        with Image.open(out/(name+'.png')) as im:im.load();qa.append({'figure':name,'PNG_decode':'passed','size':im.size,'dpi':im.info.get('dpi'),'source_rows':len(rows)})
        captions.append('**'+name+'.** '+caption);plt.close(fig)
    ratios=[r for name in ['car_rauec.csv','airfoil_rauec.csv'] for r in read(name) if primary(r) and r['implementation']=='new']
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.5),layout='constrained',sharey=True)
    for ax,(dataset,backbone,title) in zip(axes.flat,MODELS):
        rows=select(ratios,dataset,backbone);methods=[m for m in ['uniform','pi','cpg','ca'] if any(canon(r['method'])==m for r in rows)]
        for i,m in enumerate(methods):
            r=next(r for r in rows if canon(r['method'])==m);ax.errorbar(i,float(r['mean']),yerr=float(r['sampling_sd']),marker=MARKERS[m],color=COLORS[m],capsize=4)
        ax.axhline(1,color='#aaaaaa',ls='--',lw=.8);ax.set(xticks=range(len(methods)),xticklabels=[LABELS[m] for m in methods],title=title,ylabel='rAUEC (own Uniform = 1)',ylim=(.45,1.05));ax.grid(axis='y',alpha=.15)
    export(fig,'field_rauec_seed0',ratios,'Each model uses training seed0. Mean and sample SD are across sampling seeds0/1/2, after case averaging and separate matched-Uniform area ratios over37.5%,50%,75%. Each backbone has its own denominator. Conv-temp is Full-only and has no rAUEC. Car LinearNO PI/CA are unavailable in this protocol.')
    errors=[r for name in ['car_accuracy_budget.csv','airfoil_accuracy_budget.csv'] for r in read(name) if primary(r) and r['implementation']=='new']
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.8),layout='constrained')
    for ax,(dataset,backbone,title) in zip(axes.flat,MODELS):
        rows=select(errors,dataset,backbone)
        for m in ['uniform','pi','cpg','ca']:
            rs=sorted([r for r in rows if canon(r['method'])==m],key=lambda r:float(r['budget_fraction']))
            if not rs:continue
            ax.errorbar([100*float(r['budget_fraction']) for r in rs],[float(r['error_mean']) for r in rs],yerr=[float(r['sampling_sd']) for r in rs],label=LABELS[m],marker=MARKERS[m],color=COLORS[m],capsize=2,lw=1.1,markerfacecolor='white' if m=='ca' else COLORS[m])
        f=next(r for r in rows if r['method']=='full');ax.axhline(float(f['error_mean']),color='#222222',ls=':',label='Full100%');ax.set(xticks=[37.5,50,75],title=title,xlabel='Retained point budget (%)',ylabel='Absolute field error');ax.grid(alpha=.15);ax.legend(frameon=False,ncols=2)
    export(fig,'absolute_error_budget',errors,'Complete field errors use100 Car cases or110 Airfoil cases. Error bars are sample SD across the three sampling-seed case means, with one independently trained checkpoint per panel. Dotted lines are same-checkpoint Full100% references. Car uses velocity relative L2 plus0.5 pressure relative L2; Airfoil uses pressure relative L2.')
    joined=[r for r in read('accuracy_latency_summary.csv') if primary(r) and r['scope'].startswith('A') and r['implementation'] not in ['old']]
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.8),layout='constrained')
    for ax,(dataset,backbone,title) in zip(axes.flat,MODELS):
        rows=select(joined,dataset,backbone)
        for m in ['uniform','pi','cpg','ca']:
            rs=sorted([r for r in rows if canon(r['method'])==m],key=lambda r:float(r['budget_fraction']))
            if not rs:continue
            ax.plot([float(r['complete_latency_median_ms']) for r in rs],[float(r['error']) for r in rs],label=LABELS[m],marker=MARKERS[m],color=COLORS[m],lw=1.1,markerfacecolor='white' if m=='ca' else COLORS[m])
        f=next(r for r in rows if r['method']=='full');ax.scatter(float(f['complete_latency_median_ms']),float(f['error']),marker='*',s=65,color='#222222',label='Full100%')
        ax.set(title=title,xlabel='CPU → complete CPU field latency (ms)',ylabel='Absolute field error');ax.grid(alpha=.15);ax.legend(frameon=False,ncols=2)
    export(fig,'absolute_error_direct_latency',joined,'ScopeA uses directly measured CPU-input-to-complete-CPU-float64 latency, batch1 and eager FP32. Latency pools100 formal repetitions after20 warmups over3 fixed cases and3 sampling seeds; Full has no sampling seed. Accuracy uses the full100/110-case population. Each curve binds the same checkpoint and reconstruction; no old-checkpoint errors are paired with new times.')
    cost=[r for r in read('accuracy_latency_summary.csv') if r['scope'].startswith('A') and float(r['budget_fraction'])==.5 and primary(r)]
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.5),layout='constrained')
    for ax,(dataset,backbone,title) in zip(axes.flat,MODELS):
        rows=select(cost,dataset,backbone);methods=[m for m in ['uniform','pi','cpg','ca'] if any(canon(r['method'])==m for r in rows)]
        for y,m in enumerate(methods):
            old=next(r for r in rows if canon(r['method'])==m and r['implementation']=='old');new=next(r for r in rows if canon(r['method'])==m and r['implementation']=='new');x0,x1=float(old['complete_latency_median_ms']),float(new['complete_latency_median_ms'])
            ax.plot([x0,x1],[y,y],color=COLORS[m],lw=1.5);ax.scatter(x0,y,marker='x',color='#666666',s=45,label='Old' if y==0 else None);ax.scatter(x1,y,marker='o',color=COLORS[m],s=26,label='New' if y==0 else None)
        ax.set(yticks=range(len(methods)),yticklabels=[LABELS[m] for m in methods],xlabel='Complete request median (ms)',title=title);ax.set_xlim(left=0);ax.grid(axis='x',alpha=.15);ax.legend(frameon=False)
    export(fig,'old_new_complete_request_cost',cost,'Direct scopeA request medians at the fixed50% budget, retaining every available primary method. Crosses are original reconstruction and circles are the new rule-equivalent implementation. Slower Airfoil requests are retained; sparse GPU application alone does not establish an end-to-end gain.')
    cache=[]
    for r in read('car_reuse_costs.csv'):
        if float(r['budget_fraction'])!=.5:continue
        cache.append(dict(dataset='shapenet_car',backbone=r['backbone'],method=canon(r['variant']),implementation=r['implementation'],uses=int(r['uses']),derived_per_request_ms=float(r['estimated_per_request_ms']),case_id=r['case_id'],sampling_seed=r['sampling_seed']))
    for r in read('airfoil_cache_reuse_costs.csv'):
        if float(r['budget_fraction'])!=.5:continue
        cache.append(dict(dataset='airfoil2d',backbone=r['backbone'],method=r['method'],implementation=r['implementation'],uses=int(r['requests']),derived_per_request_ms=float(r['modeled_total_ms'])/int(r['requests']),case_id=r['case_id'],sampling_seed=r['sampling_seed']))
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.8),layout='constrained')
    for ax,(dataset,backbone,title) in zip(axes.flat,MODELS):
        rows=select(cache,dataset,backbone)
        for m in ['uniform','pi','cpg','ca']:
            for impl,style in [('old','--'),('new','-')]:
                rs=[r for r in rows if r['method']==m and r['implementation']==impl]
                if not rs:continue
                uses=sorted({r['uses'] for r in rs});values=[float(np.mean([r['derived_per_request_ms'] for r in rs if r['uses']==u])) for u in uses]
                ax.plot(uses,values,color=COLORS[m],ls=style,marker=MARKERS[m],markersize=3,lw=1.1,label=LABELS[m] if impl=='new' else None)
        ax.set(xscale='log',xlabel='Repeated requests with same geometry / point set',ylabel='Derived mean cost per request (ms)',title=title);ax.grid(alpha=.15);ax.legend(frameon=False,ncols=2)
    export(fig,'fixed_map_cache_reuse_cost',cache,'ScopeC at50% budget. Dashed lines use cached CPU application; solid lines use cached GPU application. Each point averages nine case/sampling-seed estimates of [measured first request+(R−1)×steady median]/R. These are derived reuse estimates, not contiguous R-request measurements. Coarse prediction, score, selection and network inference are recomputed; only the interpolation map is cached.')
    table=[]
    for dataset,backbone,title in MODELS:
        r=next(x for x in errors if x['dataset']==dataset and x['backbone']==backbone and x['method']=='full');t=next(x for x in read('accuracy_latency_summary.csv') if x['dataset']==dataset and x['backbone']==backbone and x['method']=='full' and x['scope'].startswith('A'))
        table.append({'dataset':dataset,'backbone':backbone,'training_seed':0,'full_error':r['error_mean'],'full_A_median_ms':t['complete_latency_median_ms'],'error_cases':r['case_count']})
    r=next(x for x in errors if x['backbone']=='LinearNO_conv_temp_fullgrid' and x['method']=='full');t=next(x for x in read('accuracy_latency_summary.csv') if x['backbone']=='LinearNO_conv_temp_fullgrid' and x['scope'].startswith('A'))
    table.append({'dataset':'airfoil2d','backbone':'LinearNO_conv_temp_fullgrid','training_seed':0,'full_error':r['error_mean'],'full_A_median_ms':t['complete_latency_median_ms'],'error_cases':r['case_count']})
    with (out/'full_reference_table.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    def paper_table(name,headers,cells,source,caption,width=8.2):
        fig,ax=plt.subplots(figsize=(width,2.05),layout='constrained');ax.axis('off')
        t=ax.table(cellText=cells,colLabels=headers,cellLoc='center',loc='center');t.auto_set_font_size(False);t.set_fontsize(8.5);t.scale(1,1.4)
        for (row,col),cell in t.get_celld().items():
            cell.visible_edges=''
            if row==0:cell.visible_edges='TB';cell.set_linewidth(.8);cell.get_text().set_weight('bold')
            elif row==len(cells):cell.visible_edges='B';cell.set_linewidth(.8)
        export(fig,name,source,caption)
    cells=[]
    for method in ['uniform','pi','cpg','ca']:
        row=[LABELS[method]]
        for dataset,backbone,_ in MODELS:
            rs=[r for r in ratios if r['dataset']==dataset and r['backbone']==backbone and canon(r['method'])==method]
            row.append(f"{float(rs[0]['mean']):.4f} ± {float(rs[0]['sampling_sd']):.4f}" if rs else 'Not measured')
        cells.append(row)
    paper_table('field_rauec_table',['Method','Car T3','Car LinearNO','Airfoil T3','Airfoil point-temp'],cells,ratios,
                'Field rAUEC mean ± sampling-seed sample SD for training seed0. Three matched sampling-seed ratios per model; each denominator is that backbone/checkpoint/population/reconstruction Uniform area. Car LinearNO PI/CA were not measured in this protocol.')
    labels=['Car / Transolver3','Car / LinearNO','Airfoil / Transolver3','Airfoil / point-temp','Airfoil / conv-temp Full-only']
    cells=[[label,f"{float(r['full_error']):.6f}",f"{float(r['full_A_median_ms']):.3f}",r['error_cases']] for label,r in zip(labels,table)]
    paper_table('full_reference_table',['Model','Full field error','ScopeA median (ms)','Accuracy cases'],cells,table,
                'Same-checkpoint Full100% error and CPU-to-complete-CPU-field direct latency. Each model is training seed0; error uses100 Car or110 Airfoil cases and latency uses3 fixed cases with100 formal repetitions after20 warmups. The official Airfoil conv-temp variant is Full-only. Full has no sampling seed.')
    (out/'captions.md').write_text('\n\n'.join(captions)+'\n')
    (out/'render_qa.json').write_text(json.dumps({'status':'PNG_decoded; PDF_render_pending','figures':qa},indent=2)+'\n')
    print(json.dumps(qa,indent=2))


if __name__=='__main__':main()
