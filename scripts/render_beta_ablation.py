"""Render the final beta ablation from the supplied per-seed summary."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import string
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/beta_ablation'
OUT = ROOT / 'figures/appendix/beta_ablation_assets'
BETAS = (0., .25, .5, .75, 1.)
COLORS = {'pi':'#169C9C', 'cpg':'#E45D42', 'ca':'#7543A8'}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'grid.linewidth':.4,'grid.alpha':.22,
        'lines.linewidth':1.4,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'figure.facecolor':'white'})

def save(fig,path,transparent=False,dpi=600):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(Path(str(path)+'.'+ext),dpi=dpi,bbox_inches='tight',pad_inches=.025,transparent=transparent)
        if ext=='svg':
            svg=Path(str(path)+'.svg')
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


ORDER = [('airfoil2d','field'), ('pipe','field'), ('darcy','field'), ('elasticity','field'),
         ('plasticity2d','field'), ('shapenet_car','field'), ('driverml','field'), ('superwing','field'),
         ('airfoil2d','cdcl'), ('shapenet_car','cd'), ('driverml','cd'), ('superwing','cdcl')]
NAMES = {'airfoil2d':'Airfoil2D', 'pipe':'Pipe', 'darcy':'Darcy', 'elasticity':'Elasticity',
         'plasticity2d':'Plasticity', 'shapenet_car':'ShapeNet-Car', 'driverml':'DrivAerML', 'superwing':'SuperWing'}
FONT = dict(inset_tick=8.3, tick=10.3, title=12.5, legend=11, axis=13, panel_axis=10, provisional=8.1)
X_LABEL = r'Compensation exponent $\beta$'
Y_LABEL = r'Error–budget area ratio $R_m(\beta)$'
LAYOUT = dict(figsize=(12.0,6.5),left=.073,right=.993,bottom=.155,top=.950,wspace=.26,hspace=.35)
CURVE_STYLE = {
    'pi': dict(color=COLORS['pi'], ls=(0,(6,6)), lw=2.5, marker='^', ms=7.2, mew=1.2, z=4),
    'cpg': dict(color=COLORS['cpg'], ls='-', lw=2.05, marker='s', ms=6.2, mew=1.2, z=5),
    'ca': dict(color=COLORS['ca'], ls=(0,(1.2,2.3)), lw=1.6, marker='D', ms=3.8, mew=1.25, z=6),
}
ZOOMS = {
    ('airfoil2d','field'): dict(bounds=[.58,.56,.39,.38], xlim=[.72,1.025], ylim=[.065,.285], xticks=[.75,1], yticks=[.1,.2]),
    ('pipe','field'): dict(bounds=[.58,.57,.39,.37], xlim=[.72,1.025], ylim=[.035,.235], xticks=[.75,1], yticks=[.05,.15]),
    ('airfoil2d','cdcl'): dict(bounds=[.64,.64,.34,.30], xlim=[.73,1.025], ylim=[0,.42], xticks=[.75,1], yticks=[0,.2,.4]),
}


def collect(full_only=False):
    complete=read(DATA/'summary.json')
    assert complete['status']['complete'] and complete['status']['complete_metrics']==12
    rows=[];panels=[]
    for pair in ORDER:
        selected=[dict(r) for r in complete['results'] if (r['benchmark'],r['metric'])==pair]
        assert len(selected)==15
        for method in CURVE_STYLE:
            group=[r for r in selected if r['method']==method]
            assert sorted(r['beta'] for r in group)==list(BETAS)
            for r in group:
                values=np.asarray(r['per_seed_relative_to_beta0'])
                assert values.shape==(3,) and np.isfinite(values).all()
                np.testing.assert_allclose([values.mean(),values.std(ddof=1)],[r['relative_to_beta0_mean'],r['relative_to_beta0_std']],rtol=1e-12,atol=1e-12)
                if r['beta']==0:np.testing.assert_array_equal(values,np.ones(3))
        count=selected[0]['cases']
        for r in selected:r.update(status='complete',total_cases=count,source='data/beta_ablation/summary.json')
        rows.extend(selected)
        panels.append(dict(benchmark=pair[0],metric=pair[1],status='complete',cases=count,total_cases=count,source='data/beta_ablation/summary.json',matched_sample_ids=None))
    return dict(complete=True,normalization='Per-policy, per-seed error-budget area divided by beta=0 area; then seed mean and sample standard deviation',reference_style='figures/appendix/budget_error_curves.pdf',panels=panels,results=rows)


def handle(method):
    s=CURVE_STYLE[method]
    return Line2D([],[],color=s['color'],ls=s['ls'],lw=s['lw'],marker=s['marker'],ms=s['ms'],
                  mfc='none',mew=s['mew'],label=method.upper(),dash_capstyle='round' if method=='ca' else 'butt')


def plot_series(ax, series, scale=1., paper_marks=True):
    for method, group in series:
        s=CURVE_STYLE[method]
        x=np.array([r['beta'] for r in group]);y=np.array([r['relative_to_beta0_mean'] for r in group]);sd=np.array([r['relative_to_beta0_std'] for r in group])
        ax.plot(x,y,color=s['color'],ls=s['ls'],lw=s['lw']*scale,zorder=s['z'],dash_capstyle='round' if method=='ca' else 'butt')
        ax.fill_between(x,y-sd,y+sd,color=s['color'],alpha=.08,lw=0,zorder=1)
        if paper_marks:
            r=next(r for r in group if r['beta']==r['production_beta'])
            ax.plot([r['beta']],[r['relative_to_beta0_mean']],ls='none',marker='o',ms=10.5*scale,
                    mfc='none',mec='#505860',mew=.9,zorder=18)
    for index,(method,group) in enumerate(series):
        s=CURVE_STYLE[method]
        ax.plot([r['beta'] for r in group],[r['relative_to_beta0_mean'] for r in group],ls='none',
                marker=s['marker'],ms=s['ms']*scale,mfc='none',mec=s['color'],mew=s['mew']*max(.8,scale),zorder=20+index)


def add_zoom(ax,pair,series):
    config=ZOOMS.get(pair)
    if config is None:return None
    left,bottom,width,height=config['bounds']
    # Check the full parent envelope over the region an opaque inset will cover.
    for method,group in series:
        x=np.array([r['beta'] for r in group]);y=np.array([r['relative_to_beta0_mean'] for r in group]);sd=np.array([r['relative_to_beta0_std'] for r in group])
        xs=np.linspace(x.min(),x.max(),1001);lo=np.interp(xs,x,y-sd);hi=np.interp(xs,x,y+sd)
        lower=ax.transAxes.inverted().transform(ax.transData.transform(np.c_[xs,lo]));upper=ax.transAxes.inverted().transform(ax.transData.transform(np.c_[xs,hi]))
        covered=(lower[:,0]>left)&(lower[:,0]<left+width)&(upper[:,1]>bottom)&(lower[:,1]<bottom+height)
        if covered.any():raise ValueError(f'Inset would obscure uncertainty envelope: {pair}/{method}')
    inset=ax.inset_axes(config['bounds'],zorder=30)
    inset.set_facecolor('white');plot_series(inset,series,scale=.60,paper_marks=False)
    inset.set_xlim(config['xlim']);inset.set_ylim(config['ylim']);inset.set_xticks(config['xticks']);inset.set_yticks(config['yticks'])
    inset.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));inset.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    inset.tick_params(labelsize=FONT['inset_tick'],length=2,width=.5,pad=1)
    for sp in inset.spines.values():sp.set_visible(True);sp.set_color('#818991');sp.set_linewidth(.55)
    inset.grid(alpha=.15,lw=.3)
    ax.indicate_inset_zoom(inset,edgecolor='#818991',lw=.55,alpha=.65,zorder=2)
    return dict(**config,parent_envelope_overlap=False)


def main(full_only=False):
    data=collect(full_only);OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    write(OUT/'data/summary.json',data)
    fields=['benchmark','metric','status','cases','total_cases','method','beta','production_beta','rauec_mean','rauec_std','relative_to_beta0_mean','relative_to_beta0_std']
    with (OUT/'data/summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(data['results'])
    style();plt.rcParams['lines.scale_dashes']=False
    fig,axes=plt.subplots(3,4,figsize=LAYOUT['figsize'])
    fig.subplots_adjust(**{k:v for k,v in LAYOUT.items() if k!='figsize'})
    checks=[]
    def draw(ax,pair,title,local=False):
        coverage=next(p for p in data['panels'] if (p['benchmark'],p['metric'])==pair)
        series=[(m,sorted([r for r in data['results'] if (r['benchmark'],r['metric'],r['method'])==(*pair,m)],key=lambda r:r['beta'])) for m in CURVE_STYLE]
        lows=[1.];highs=[1.]
        for _,group in series:
            lows.extend(r['relative_to_beta0_mean']-r['relative_to_beta0_std'] for r in group)
            highs.extend(r['relative_to_beta0_mean']+r['relative_to_beta0_std'] for r in group)
        low,high=min(lows),max(highs);span=max(high-low,.025)
        bottom=low-.10*span if low<0 else max(0,low-.10*span);top=high+.13*span
        ax.set_ylim(bottom,top);ax.set_xlim(-.035,1.035)
        ticks=[v for v in MaxNLocator(4,min_n_ticks=3).tick_values(bottom,top) if bottom<=v<=top and abs(v-1)>.12*(top-bottom)]
        if pair==('airfoil2d','cdcl'):
            ticks=[v for v in [-2,1,3,6] if bottom<=v<=top]
        ticks=sorted(set([*ticks,1.]));ax.yaxis.set_major_locator(FixedLocator(ticks));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.3g}'))
        ax.set_xticks(BETAS,['0','0.25','0.5','0.75','1'])
        ax.tick_params(labelsize=FONT['tick']);ax.grid(alpha=.2)
        ax.axhline(1,color='#303840',lw=1.1,ls=(0,(2,2)),zorder=2)
        plot_series(ax,series)
        ax.text(0,1.035,title,transform=ax.transAxes,ha='left',va='bottom',fontsize=FONT['title'])
        for label,value in zip(ax.get_yticklabels(),ticks):
            if value==1:label.set_weight('bold')
        if coverage['status']!='complete':
            ax.text(.025,.025,f"Partial: {coverage['cases']:,}/{coverage['total_cases']:,}",transform=ax.transAxes,
                    ha='left',va='bottom',fontsize=FONT['provisional'],color='#505860',
                    bbox=dict(facecolor='white',edgecolor='none',pad=.6,alpha=.9),zorder=40)
        if local:
            ax.set_xlabel(X_LABEL,fontsize=FONT['panel_axis']);ax.set_ylabel(Y_LABEL,fontsize=FONT['panel_axis'])
        zoom=add_zoom(ax,pair,series)
        return dict(**coverage,y_limits=[bottom,top],scale='linear',zoom=zoom,all_beta_values_present=True,
                    whole_standard_deviation_envelope_visible=True,axis_bounds=list(ax.get_position().bounds))
    for i,(pair,ax) in enumerate(zip(ORDER,axes.flat)):
        suffix=(' · p' if pair==('driverml','field') else '' if pair[1]=='field' else ' · '+{'cd':'Cd','cdcl':'Cd/Cl'}[pair[1]])
        title=NAMES[pair[0]]+suffix
        checks.append(draw(ax,pair,f'({string.ascii_lowercase[i]}) '+title))
        single,sa=plt.subplots(figsize=(3.7,2.7));single.subplots_adjust(left=.205,right=.985,bottom=.21,top=.87)
        draw(sa,pair,title,True);save(single,OUT/'panels'/f'{pair[0]}_{pair[1]}',dpi=500)
    handles=[handle(m) for m in CURVE_STYLE]
    handles.extend([Line2D([],[],color='#303840',lw=1.1,ls=(0,(2,2)),label=r'No correction ($\beta=0$)'),
                    Line2D([],[],ls='none',marker='o',mfc='none',mec='#505860',ms=9,mew=.9,label=r'Sampling $\beta$')])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.010),ncol=5,frameon=False,fontsize=FONT['legend'],columnspacing=1.45)
    fig.supxlabel(X_LABEL,y=.082,fontsize=FONT['axis']);fig.supylabel(Y_LABEL,x=.012,fontsize=FONT['axis'])
    save(fig,OUT/'Fig_beta_ablation_overview',dpi=600)
    legend=plt.figure(figsize=(8.8,.45));legend.legend(handles=handles,loc='center',ncol=5,frameon=False,fontsize=FONT['legend'])
    save(legend,OUT/'legends/methods',transparent=True,dpi=400)
    from PIL import Image
    with Image.open(OUT/'Fig_beta_ablation_overview.png') as picture:
        picture.thumbnail((2300,1700));picture.save(OUT/'preview.png')
    write(OUT/'qa.json',dict(complete=data['complete'],panels=checks,font_sizes_pt=FONT,curve_styles=CURVE_STYLE,
                             layout=LAYOUT,marker_positions_offset=False,x_label=X_LABEL,y_label=Y_LABEL,
                             normalization=data['normalization'],uncertainty='One sample SD across three paired seeds'))
    for ext in ['png','pdf','svg']:
        shutil.copyfile(OUT/f'Fig_beta_ablation_overview.{ext}',ROOT/f'figures/appendix/beta_ablation.{ext}')
    report(data)
    print('Rendered 12-metric overview:',OUT/'Fig_beta_ablation_overview.pdf',flush=True)
    print('Complete metrics:',sum(p['status']=='complete' for p in data['panels']),'; provisional:',
          [(p['benchmark'],p['metric'],p['cases'],p['total_cases']) for p in data['panels'] if p['status']!='complete'],flush=True)


def report(data):
    lines=['# β ablation overview','',f"All metrics complete: {data['complete']}.",'',
           'Figure style follows the supplied appendix budget figure. Both axes use a metric name followed by its mathematical symbol: "Compensation exponent β" and "Error–budget area ratio R_m(β)". For policy m and seed s, let A_m,s(beta) be the trapezoidal error-budget area over 37.5%, 50%, and 75% budgets. The displayed value is R_m(beta) = mean_s[A_m,s(beta) / A_m,s(0)], so the denominator is the same policy and seed at beta=0. Bands show one sample standard deviation of these ratios across the three sampling seeds. The dashed reference is no measure correction; gray rings mark the unchanged paper beta.','',
           'Provisional panels use a matched intersection of available geometries across all methods, beta values, seeds and budgets. Their values can change as the experiment finishes.','',
           '| Dataset | Metric | Status | Cases | PI rAUEC | CPG rAUEC | CA rAUEC |','|---|---|---|---:|---:|---:|---:|']
    for p in data['panels']:
        values=[next(r['rauec_mean'] for r in data['results'] if (r['benchmark'],r['metric'],r['method'])==(p['benchmark'],p['metric'],m) and r['beta']==r['production_beta']) for m in CURVE_STYLE]
        lines.append(f"| {NAMES[p['benchmark']]} | {p['metric']} | {p['status']} | {p['cases']}/{p['total_cases']} | "+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['','The table uses rAUEC relative to matched Uniform at the paper beta; it has a common reference across methods. The figure uses each policy’s own beta=0 reference.','',
            'Files: `Fig_beta_ablation_overview.png/pdf/svg`; `panels/` and `legends/` contain individual assets; `data/summary.json/csv` contains the displayed values.','',
            'Reproduce this figure with python3 scripts/render_beta_ablation.py from the repository root.']
    if data['complete']:
        lines=[('All twelve metrics use their complete planned test populations. There are no provisional panels.'
                if line.startswith('Provisional panels use') else
                'Reproduce this figure with python3 scripts/render_beta_ablation.py from the repository root.'
                if line.startswith('The native finalizer regenerates') else line) for line in lines]
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full-only',action='store_true');args=parser.parse_args()
    main(full_only=args.full_only)
