"""Deterministic paper plots/tables from audited Car CSVs only."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'uniform':'#666666','pi':'#E69F00','cpg_beta0':'#56B4E9','cpg_beta1':'#0072B2','ca':'#009E73'}
LABELS={'full':'Full','uniform':'Uniform','pi':'PI','cpg_beta0':r'CPG $\beta=0$','cpg_beta1':r'CPG $\beta=1$','ca':'CA'}
ARCHES=['Transolver3','LinearNO'];METHODS=['uniform','pi','cpg_beta1','ca'];BUDGETS=[.375,.5,.75]


def read_csv(path):return list(csv.DictReader(Path(path).open(newline='')))
def number(row,key):return float(row[key]) if row[key] not in ['N/A','','None'] else None
def save(fig,root,name):
    for extension in ['pdf','svg','png']:fig.savefig(root/f'{name}.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def polish(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)

def lookup(rows,**criteria):
    matches=[row for row in rows if all(row.get(k)==str(v) for k,v in criteria.items())]
    assert len(matches)==1,(criteria,len(matches));return matches[0]

def fmt(value,precision=4):return 'N/A' if value is None else f'{value:.{precision}g}'

def make_tables(root,budgets,rauec,latency):
    tables=root/'tables';tables.mkdir(exist_ok=True)
    field=[r for r in rauec if r['metric']=='field_error']
    for impl in ['old','new']:
        rows=[];latex=[r'\begin{tabular}{llc}',r'\toprule',r'Backbone & Method & ShapeNet-Car Field \\',r'\midrule']
        for arch in ARCHES:
            available=[r for r in field if r['backbone']==arch and r['implementation']==impl and r['variant'] in METHODS]
            ranks=sorted(set(round(float(r['rAUEC_mean']),3) for r in available))
            for method in METHODS:
                candidates=[r for r in available if r['variant']==method]
                if candidates:
                    r=candidates[0];mean=float(r['rAUEC_mean']);sd=float(r['rAUEC_sampling_sd']);cell=f'{mean:.3f} ± {sd:.3f}'
                    rank=ranks.index(round(mean,3));value=f'{mean:.3f}'
                    value=(r'\textbf{'+value+'}') if rank==0 else (r'\underline{'+value+'}') if rank==1 else value
                    tex=r'\shortstack{'+value+r'\\ $\pm '+f'{sd:.3f}'+r'$}'
                else:cell='N/A';tex='N/A'
                display=LABELS[method].replace('$','').replace('\\beta',r'\beta')
                if method=='cpg_beta1':display=r'CPG $\beta=1$'
                rows.append(f'| {arch} | {method} | {cell} |')
                latex.append(f'{arch} & {display} & {tex} '+r'\\')
            if arch!=ARCHES[-1]:latex.append(r'\midrule')
        latex.extend([r'\bottomrule',r'\end{tabular}'])
        (tables/f'car_rauec_{impl}.tex').write_text('\n'.join(latex)+'\n')
        note='Sampling seeds n=3, sample SD (ddof=1). Each backbone uses its own matched Uniform denominator. Full is excluded. Best/second are marked only within each backbone; ties at displayed precision share rank. LinearNO PI/CA are unavailable in this retained historical Car matrix.\n'
        (tables/f'car_rauec_{impl}.md').write_text(f'Car {impl} reconstruction rAUEC\n\n| Backbone | Method | Field rAUEC |\n|---|---|---|\n'+'\n'.join(rows)+'\n\n'+note)
    beta=['| Backbone | Reconstruction | CPG beta=0 | CPG beta=1 |','|---|---|---|---|']
    for arch in ARCHES:
        for impl in ['old','new']:
            cells=[]
            for variant in ['cpg_beta0','cpg_beta1']:
                row=lookup(field,backbone=arch,implementation=impl,variant=variant)
                cells.append(f"{float(row['rAUEC_mean']):.4f} ± {float(row['rAUEC_sampling_sd']):.4f}")
            beta.append(f'| {arch} | {impl} | {cells[0]} | {cells[1]} |')
    (tables/'car_beta_mechanism.md').write_text('\n'.join(beta)+'\n\nSame retained-index and coarse-prediction hashes are required within each beta pair.\n')
    for scope in ['A','B','C']:
        lines=['| Backbone | Method | Budget | Old recon ms | New recon ms | Old total ms | New total ms | New p95 ms | New peak allocated MiB |',
               '|---|---|---|---|---|---|---|---|---|']
        for arch in ARCHES:
            variants=['full']+[m for m in METHODS if any(r['backbone']==arch and r['variant']==m for r in latency)]
            for method in variants:
                for budget in ([1.] if method=='full' else BUDGETS):
                    shared=method=='full';old=lookup(latency,scope=scope,backbone=arch,variant=method,budget_fraction=budget,implementation='shared' if shared else 'old')
                    new=old if shared else lookup(latency,scope=scope,backbone=arch,variant=method,budget_fraction=budget,implementation='new')
                    values=[float(r[k]) for r,k in [(old,'reconstruction_ms'),(new,'reconstruction_ms'),(old,'direct_total_ms'),(new,'direct_total_ms'),(new,'p95_ms')]]
                    lines.append(f"| {arch} | {method} | {budget:.1%} | "+' | '.join(f'{x:.2f}' for x in values)+f" | {float(new['peak_memory_allocated_bytes'])/2**20:.1f} |")
        note='Reconstruction denotes the separately instrumented complete output pipeline (decode, mapping, application, conversion and required transfers). Direct total is measured without stage insertion. Latency is pooled over100 formal repeats at each of3 fixed cases and3 sampling seeds; Full has3 cases and one shared old/new series. Peak covers the PyTorch allocator; external transient GPU allocation is not fully observed.'
        (tables/f'car_efficiency_{scope}.md').write_text(f'Scope {scope}\n\n'+'\n'.join(lines)+'\n\n'+note+'\n')
    lines=['| Backbone | Reconstruction | Method | Full 100% | 37.5% | 50% | 75% |','|---|---|---|---|---|---|---|']
    for arch in ARCHES:
        for impl in ['old','new']:
            full=lookup(budgets,backbone=arch,implementation=impl,variant='full',metric='field_error',budget_fraction=1.)
            for method in METHODS:
                values=[r for r in budgets if r['backbone']==arch and r['implementation']==impl and r['variant']==method and r['metric']=='field_error']
                if not values:continue
                cells=[]
                for budget in BUDGETS:
                    r=lookup(values,budget_fraction=budget);cells.append(f"{float(r['error_mean']):.5f} ± {float(r['sampling_sd']):.5f}")
                lines.append(f"| {arch} | {impl} | {method} | {float(full['error_mean']):.5f} | "+' | '.join(cells)+' |')
    (tables/'car_absolute_errors.md').write_text('\n'.join(lines)+'\n\nCar Field = velocity relative L2 +0.5×surface-pressure relative L2 on all100 historical test cases. Error bars are sampling SD, not training SD.\n')


def generate(root):
    root=Path(root);figdir=root/'figures';figdir.mkdir(exist_ok=True)
    budgets=read_csv(root/'results/car_budget_metrics.csv');rauec=read_csv(root/'results/car_rauec.csv');latency=read_csv(root/'results/car_latency_grouped.csv')
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,
                         'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'lines.linewidth':1.6})
    # Separate old/new panels avoid treating execution modes as sampling methods.
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.9),sharey=True)
    field=[r for r in rauec if r['metric']=='field_error']
    for ax,impl,title in zip(axes,['old','new'],['(a) Native reconstruction','(b) Accepted sparse application']):
        for ai,arch in enumerate(ARCHES):
            for mi,method in enumerate(METHODS):
                rows=[r for r in field if r['backbone']==arch and r['implementation']==impl and r['variant']==method]
                if not rows:continue
                r=rows[0];x=ai+(mi-1.5)*.17
                ax.errorbar(x,float(r['rAUEC_mean']),yerr=float(r['rAUEC_sampling_sd']),fmt='o',ms=5,capsize=2,color=COLORS[method])
        ax.axhline(1.,ls='--',lw=.8,color='#333333');ax.set_xticks([0,1],ARCHES);ax.set_xlim(-.45,1.45);ax.set_title(title);polish(ax)
    axes[0].set_ylabel('Field rAUEC (lower is better)')
    fig.legend([Line2D([],[],color=COLORS[m],marker='o',ls='') for m in METHODS],[LABELS[m] for m in METHODS],loc='lower center',ncol=4,bbox_to_anchor=(.5,-.04),frameon=False)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,figdir,'car_rauec')
    for impl in ['old','new']:
        fig,axes=plt.subplots(1,2,figsize=(7.4,2.9),sharey=True)
        for ax,arch in zip(axes,ARCHES):
            for method in METHODS:
                rows=[r for r in budgets if r['backbone']==arch and r['implementation']==impl and r['variant']==method and r['metric']=='field_error']
                if not rows:continue
                rows=sorted(rows,key=lambda r:float(r['budget_fraction']));ax.errorbar([100*float(r['budget_fraction']) for r in rows],[float(r['error_mean']) for r in rows],yerr=[float(r['sampling_sd']) for r in rows],color=COLORS[method],marker='o',ms=4,capsize=2,label=LABELS[method])
            full=lookup(budgets,backbone=arch,implementation=impl,variant='full',metric='field_error',budget_fraction=1.)
            ax.axhline(float(full['error_mean']),color='#222222',ls='--',lw=1,label='Full 100%');ax.set_title(arch);ax.set_xticks([37.5,50,75]);ax.set_xlabel('Retained budget (%)');polish(ax)
        axes[0].set_ylabel('Complete-field error');axes[0].legend(frameon=False,fontsize=7);fig.tight_layout();save(fig,figdir,f'car_absolute_error_{impl}')
    for scope in ['A','B','C']:
        fig,axes=plt.subplots(2,2,figsize=(7.4,5.0),sharex=True)
        for ai,arch in enumerate(ARCHES):
            for ci,metric in enumerate(['reconstruction_ms','direct_total_ms']):
                ax=axes[ai,ci]
                for method in METHODS:
                    for impl,style in [('old','--'),('new','-')]:
                        rows=[r for r in latency if r['scope']==scope and r['backbone']==arch and r['variant']==method and r['implementation']==impl]
                        if not rows:continue
                        rows=sorted(rows,key=lambda r:float(r['budget_fraction']));ax.plot([100*float(r['budget_fraction']) for r in rows],[float(r[metric]) for r in rows],ls=style,color=COLORS[method],marker='o',ms=3)
                if ci==1:
                    full=lookup(latency,scope=scope,backbone=arch,variant='full',implementation='shared',budget_fraction=1.)
                    ax.axhline(float(full[metric]),ls=':',color='#222222',lw=1)
                ax.set_title(f'{arch}: '+('reconstruction' if ci==0 else 'complete request'));ax.set_ylabel('Time (ms)');ax.set_xticks([37.5,50,75]);polish(ax)
        for ax in axes[1]:ax.set_xlabel('Retained budget (%)')
        handles=[Line2D([],[],color=COLORS[m]) for m in METHODS]+[Line2D([],[],color='#222222',ls=s) for s in ['--','-',':']]
        fig.legend(handles,[LABELS[m] for m in METHODS]+['Old','New','Full 100%'],ncol=7,loc='lower center',bbox_to_anchor=(.5,-.01),frameon=False)
        fig.tight_layout(rect=(0,.04,1,1));save(fig,figdir,f'car_old_new_{scope}')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.0),sharex=True,sharey=True)
    for ax,impl in zip(axes,['old','new']):
        for arch,marker,style in [('Transolver3','o','-'),('LinearNO','s','--')]:
            for method in METHODS:
                ys=[];xs=[]
                for budget in BUDGETS:
                    rows=[r for r in latency if r['scope']=='A' and r['backbone']==arch and r['variant']==method and r['implementation']==impl and float(r['budget_fraction'])==budget]
                    if not rows:continue
                    error=lookup(budgets,backbone=arch,implementation=impl,variant=method,metric='field_error',budget_fraction=budget)
                    xs.append(float(rows[0]['direct_total_ms']));ys.append(float(error['error_mean']))
                if xs:ax.plot(xs,ys,color=COLORS[method],marker=marker,ls=style,ms=4)
            fulltime=lookup(latency,scope='A',backbone=arch,variant='full',implementation='shared',budget_fraction=1.)
            fullerror=lookup(budgets,backbone=arch,implementation=impl,variant='full',metric='field_error',budget_fraction=1.)
            ax.scatter([float(fulltime['direct_total_ms'])],[float(fullerror['error_mean'])],marker=marker,color='#222222',s=28)
        ax.set_title('Native' if impl=='old' else 'Accepted optimization');ax.set_xlabel('Complete CPU→CPU request (ms)');polish(ax)
    axes[0].set_ylabel('Complete-field error')
    handles=[Line2D([],[],color=COLORS[m]) for m in METHODS]+[Line2D([],[],color='#222222',marker=m,ls=s) for m,s in [('o','-'),('s','--')]]
    fig.legend(handles,[LABELS[m] for m in METHODS]+ARCHES,ncol=6,loc='lower center',bbox_to_anchor=(.5,-.05),frameon=False)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,figdir,'car_error_latency_A')
    make_tables(root,budgets,rauec,latency)
    (figdir/'captions.md').write_text('''Car rAUEC: the mean and sample SD over3 sampling seeds. Each backbone is divided by its own matched Uniform AUEC. Full100% is excluded; CPG uses the pre-fixed beta=1. Native and optimized reconstruction are separate panels.\n\nAbsolute error: all100 historical Car test cases; Field = velocity relative L2 +0.5×surface-pressure relative L2. Each backbone has its own same-checkpoint Full reference.\n\nEfficiency: scopes A, B and C appear in separate figures. A is CPU input→complete CPU float64 field. B has resident GPU model inputs→complete GPU float64 field and includes CPU auxiliary work. C is CPU→CPU with only the explicitly keyed interpolation map reused. Dashed/solid are old/new execution. Reconstruction is a separate instrumented output-pipeline measurement. Complete request medians come from uninstrumented synchronized wall timing. Full uses one measured series shared by old/new.\n\nError–latency: absolute errors use100 cases; latency uses the3 predeclared representative cases and3 sampling seeds,100 formal repeats each. This is a comparison of accuracy and representative latency, not a100-case latency census. No rAUEC is used as latency.\n''')


if __name__=='__main__':generate(Path(__file__).resolve().parents[1])
