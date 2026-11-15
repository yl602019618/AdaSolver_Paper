#!/usr/bin/env python3
"""CSV-only P0 figures; uncertainty and GPU blocks remain explicit."""
import argparse,csv,hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

MODELS=['Transolver3','LinearNO_point_temp']
METHODS=['uniform','pi','cpg_beta0','cpg','ca']
LABELS={'uniform':'Uniform','pi':'PI','cpg_beta0':'CPG beta0','cpg':'CPG beta1','ca':'CA','full':'Full'}
COLORS={'uniform':'#666666','pi':'#0072B2','cpg_beta0':'#E69F00','cpg':'#D55E00','ca':'#009E73','full':'#222222'}
MARKERS={'uniform':'o','pi':'s','cpg_beta0':'v','cpg':'^','ca':'D','full':'*'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();d=a.data_dir;out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    def read(name):
        path=d/name
        if not path.exists():return []
        with path.open() as f:return list(csv.DictReader(f))
    verification=json.loads((d/'verification.json').read_text());complete=verification['complete']
    train=[r for r in read('training_seed_summary.csv') if r['implementation']=='new' and r['metric']=='field_relative_l2']
    ratio=[r for r in train if r['quantity']=='rAUEC'];error=[r for r in train if r['quantity']=='absolute_error']
    latency=[r for r in read('backbone_accuracy_latency.csv') if r['scope']=='A_cpu_cpu']
    seed_ids=sorted({int(r['training_seed']) for r in latency})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'airfoil-p0-three-seeds-v1','savefig.dpi':300})
    qa=[];captions=[];script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def export(fig,name,rows,caption):
        for ext in ['pdf','svg','png']:
            meta={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {}
            fig.savefig(out/(name+'.'+ext),bbox_inches='tight',pad_inches=.06,metadata=meta)
        with (out/(name+'_source.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)
        with Image.open(out/(name+'.png')) as im:im.load();entry=dict(figure=name,PNG_decode='passed',size=im.size,dpi=im.info.get('dpi'),source_rows=len(rows),script_sha256=script_sha)
        if shutil.which('pdftoppm'):
            import tempfile
            with tempfile.TemporaryDirectory(prefix='airfoil_plot_qa_') as temp:
                subprocess.run(['pdftoppm','-f','1','-singlefile','-r','80','-png',str(out/(name+'.pdf')),str(Path(temp)/'page')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                with Image.open(Path(temp)/'page.png') as im:im.load()
            entry['PDF_render']='passed'
        qa.append(entry);captions.append('**'+name+'.** '+caption);plt.close(fig)
    status_note='All three independent training seeds are included.' if complete else 'This is a partial snapshot; missing training seeds or timing conditions are not filled.'
    if ratio:
        fig,axes=plt.subplots(1,2,figsize=(8.5,3.25),layout='constrained',sharey=True)
        for ax,model in zip(axes,MODELS):
            rows=[r for r in ratio if r['backbone']==model]
            for i,method in enumerate(METHODS):
                found=[r for r in rows if r['method']==method]
                if not found:continue
                r=found[0];values=[(s,float(r[f'seed{s}'])) for s in [0,1,2] if r[f'seed{s}']]
                for s,value in values:ax.scatter(i+(s-1)*.1,value,s=18,color=COLORS[method],alpha=.65)
                yerr=float(r['sample_sd']) if r['sample_sd'] else None
                ax.errorbar(i,float(r['mean']),yerr=yerr,marker='_',markersize=12,color='black',capsize=4,zorder=4)
            ax.axhline(1,color='#aaaaaa',ls='--',lw=.8);ax.set(xticks=range(5),xticklabels=['Uniform','PI','CPG\nbeta0','CPG\nbeta1','CA'],title=model,ylabel='rAUEC (own Uniform = 1)');ax.grid(axis='y',alpha=.15)
        export(fig,'training_seed_rauec',ratio,'Dots are individual training-model rAUECs after each model aggregates its three matched sampling-seed area ratios. Black marks show the mean; sample-SD bars are drawn only after three independent training models exist. Beta1 is primary and beta0 is the paired compensation ablation. '+status_note)
    if error:
        fig,axes=plt.subplots(1,2,figsize=(8.5,3.25),layout='constrained')
        for ax,model in zip(axes,MODELS):
            rows=[r for r in error if r['backbone']==model]
            for method in METHODS:
                rs=sorted([r for r in rows if r['method']==method],key=lambda r:float(r['budget']))
                if not rs:continue
                x=[100*float(r['budget']) for r in rs];y=[float(r['mean']) for r in rs]
                if all(r['sample_sd'] for r in rs):ax.errorbar(x,y,yerr=[float(r['sample_sd']) for r in rs],color=COLORS[method],marker=MARKERS[method],capsize=3,label=LABELS[method],lw=1.2)
                else:ax.plot(x,y,color=COLORS[method],marker=MARKERS[method],label=LABELS[method],lw=1.2)
            full=[r for r in rows if r['method']=='full']
            if full:ax.axhline(float(full[0]['mean']),color='#222222',ls=':',label='Full100%')
            ax.set(xticks=[37.5,50,75],xlabel='Retained point budget (%)',ylabel='Pressure relative L2 error',title=model);ax.grid(alpha=.15);ax.legend(frameon=False,ncols=2)
        export(fig,'training_seed_error_budget',error,'Case means and sampling means are computed within each checkpoint before averaging independent training models. Error bars are the sample SD across three models when complete. The dotted Full reference uses the same independently trained checkpoints at100% points. '+status_note)
    if latency:
        fig,axes=plt.subplots(len(seed_ids),2,figsize=(8.6,3.15*len(seed_ids)),layout='constrained',squeeze=False)
        source=[]
        for row,seed in enumerate(seed_ids):
            for ax,model in zip(axes[row],MODELS):
                rs=[r for r in latency if int(r['training_seed'])==seed and r['backbone']==model and r['implementation']!='old'];source+=rs
                for method in METHODS:
                    chosen=sorted([r for r in rs if r['method']==method],key=lambda r:float(r['budget_fraction']))
                    if chosen:ax.plot([float(r['condition_median_mean_ms']) for r in chosen],[float(r['error']) for r in chosen],color=COLORS[method],marker=MARKERS[method],label=LABELS[method],lw=1.2)
                full=[r for r in rs if r['method']=='full']
                if full:ax.scatter(float(full[0]['condition_median_mean_ms']),float(full[0]['error']),marker='*',color='#222222',s=70,label='Full100%')
                gpu=rs[0]['physical_gpu'] if rs else 'pending'
                ax.set(title=f'{model} · training seed{seed} · GPU{gpu}',xlabel='Complete CPU field latency (ms)',ylabel='Pressure relative L2 error');ax.grid(alpha=.15);ax.legend(frameon=False,ncols=2)
        export(fig,'checkpoint_error_latency',source,'Each panel is one checkpoint and physical GPU. ScopeA latency averages formal condition medians over the fixed3-case/3-sampling-seed matrix;100 formal repetitions and20 warmups are retained per condition. Accuracy uses110 cases under the same checkpoint and reconstruction. No raw timings are pooled across training seeds or GPUs. '+status_note)
        fig,axes=plt.subplots(len(seed_ids),2,figsize=(8.6,3.05*len(seed_ids)),layout='constrained',squeeze=False);source=[]
        for row,seed in enumerate(seed_ids):
            for ax,model in zip(axes[row],MODELS):
                rs=[r for r in latency if int(r['training_seed'])==seed and r['backbone']==model and float(r['budget_fraction'])==.5];source+=rs
                labels=[]
                for method in METHODS:
                    pair={r['implementation']:r for r in rs if r['method']==method}
                    if not {'old','new'}<=set(pair):continue
                    pos=len(labels);labels.append(LABELS[method]);x0=float(pair['old']['condition_median_mean_ms']);x1=float(pair['new']['condition_median_mean_ms'])
                    ax.plot([x0,x1],[pos,pos],color=COLORS[method],lw=1.3);ax.scatter(x0,pos,marker='x',color='#666666',s=40,label='Old' if pos==0 else None);ax.scatter(x1,pos,color=COLORS[method],s=25,label='New' if pos==0 else None)
                gpu=rs[0]['physical_gpu'] if rs else 'pending'
                ax.set(yticks=range(len(labels)),yticklabels=labels,title=f'{model} · seed{seed} · GPU{gpu}',xlabel='Complete request median mean (ms)');ax.set_xlim(left=0);ax.grid(axis='x',alpha=.15);ax.legend(frameon=False)
        export(fig,'old_new_checkpoint_cost',source,'ScopeA at the fixed50% budget, separated by checkpoint/GPU. Crosses and circles compare original and rule-equivalent new reconstruction. All available methods, including beta0 and slower outcomes, remain visible. '+status_note)
    cache=read('cache_reuse.csv')
    if cache:
        fig,axes=plt.subplots(len(seed_ids),2,figsize=(8.6,3.1*len(seed_ids)),layout='constrained',squeeze=False);source=[]
        for row,seed in enumerate(seed_ids):
            for ax,model in zip(axes[row],MODELS):
                rs=[r for r in cache if int(r['training_seed'])==seed and r['backbone']==model and float(r['budget_fraction'])==.5];source+=rs
                for method in METHODS:
                    for impl,style in [('old','--'),('new','-')]:
                        chosen=[r for r in rs if r['method']==method and r['implementation']==impl]
                        if not chosen:continue
                        uses=sorted({int(r['uses']) for r in chosen});values=[float(np.mean([float(r['derived_per_request_ms']) for r in chosen if int(r['uses'])==n])) for n in uses]
                        ax.plot(uses,values,color=COLORS[method],ls=style,marker=MARKERS[method],markersize=3,label=LABELS[method] if impl=='new' else None)
                ax.set(xscale='log',xlabel='Same-geometry / point-set requests',ylabel='Derived per-request cost (ms)',title=f'{model} · training seed{seed}');ax.grid(alpha=.15);ax.legend(frameon=False,ncols=2)
        export(fig,'checkpoint_cache_reuse',source,'ScopeC retains the measured first request and derives [first+(R−1)×steady median]/R for each case/sampling condition. Dashed CPU-map and solid GPU-map curves average these estimates only within one checkpoint. They are not contiguous R-request wall-time measurements. '+status_note)
    if ratio:
        cells=[]
        for method in METHODS:
            row=[LABELS[method]]
            for model in MODELS:
                found=[r for r in ratio if r['method']==method and r['backbone']==model]
                if not found:row.append('Pending')
                else:
                    r=found[0];row.append(f"{float(r['mean']):.4f} ± {float(r['sample_sd']):.4f}" if r['sample_sd'] else f"{float(r['mean']):.4f} (n={r['independent_training_count']}; SD pending)")
            cells.append(row)
        fig,ax=plt.subplots(figsize=(7.2,2.05),layout='constrained');ax.axis('off');table=ax.table(cellText=cells,colLabels=['Method','Transolver3','LinearNO point-temp'],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(8.5);table.scale(1,1.35)
        for (row,col),cell in table.get_celld().items():
            cell.visible_edges=''
            if row==0:cell.visible_edges='TB';cell.set_linewidth(.8);cell.get_text().set_weight('bold')
            elif row==len(cells):cell.visible_edges='B';cell.set_linewidth(.8)
        export(fig,'training_seed_rauec_table',ratio,'rAUEC mean ± sampleSD across three independently trained models after within-model sampling aggregation. Pending SD is not substituted with zero. '+status_note)
    (out/'captions.md').write_text('\n\n'.join(captions)+'\n')
    (out/'render_qa.json').write_text(json.dumps({'status':'passed' if all(r.get('PDF_render')=='passed' for r in qa) else 'partial','figures':qa,'data_status':verification['status'],'no_cross_GPU_pooling':True},indent=2)+'\n')
    print(json.dumps({'figures':len(qa),'data_status':verification['status'],'rendered_PDFs':sum(r.get('PDF_render')=='passed' for r in qa)},indent=2))


if __name__=='__main__':main()
