"""Plot the independently verified standard-cohort training-seed rAUEC CSV."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'uniform':'#666C73','pi':'#169C9C','cpg':'#E45D42','ca':'#7543A8'}
METHODS=['uniform','pi','cpg','ca']
DATASETS=['elasticity','pipe','darcy']


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).with_name('training_seed_rauec.csv'));p.add_argument('--output-dir',type=Path,default=Path(__file__).with_name('figures'));args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with args.data.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==12
    source={(r['dataset'],r['method']):r for r in rows};assert len(source)==12
    for row in rows:
        values=np.array([float(row[f'training_seed{s}']) for s in [0,1,2]])
        assert np.allclose([values.mean(),values.std(ddof=1)],[float(row['mean']),float(row['training_seed_sd'])],atol=1e-14,rtol=1e-12)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'E5-standard-training-seeds-20261115','xtick.labelsize':10,'ytick.labelsize':9})
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.7),sharey=True)
    fig.subplots_adjust(left=.075,right=.988,bottom=.24,top=.80,wspace=.12)
    for ax,dataset,label in zip(axes,DATASETS,['(a) Elasticity','(b) Pipe','(c) Darcy']):
        ax.set_title(label,pad=11,fontweight='semibold')
        ax.axhline(1.,color='#AAAAAA',linestyle=(0,(4,3)),linewidth=.8,zorder=0)
        ax.grid(axis='y',color='#E6E6E6',linewidth=.6,zorder=0)
        for x,method in enumerate(METHODS):
            r=source[dataset,method];values=[float(r[f'training_seed{s}']) for s in [0,1,2]]
            ax.scatter(x+np.array([-.16,-.06,.04]),values,s=23,facecolors='white',edgecolors='#606060',linewidths=.75,zorder=3)
            ax.errorbar(x+.18,float(r['mean']),yerr=float(r['training_seed_sd']),fmt='o',color=COLORS[method],mfc=COLORS[method],mec='white',mew=.5,markersize=6.5,elinewidth=1.3,capsize=3,capthick=1.1,zorder=4)
        ax.set_xlim(-.42,3.47);ax.set_ylim(.37,1.055)
        ax.set_xticks(np.arange(4),['Uniform','PI','CPG','CA']);ax.set_yticks([.4,.6,.8,1.])
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both',length=3,width=.7)
    axes[0].set_ylabel('Field rAUEC (lower is better)')
    fig.text(.5,.95,'Three independently trained Transolver3 models per dataset',ha='center',va='top',fontsize=12)
    handles=[Line2D([],[],linestyle='',marker='o',markerfacecolor='white',markeredgecolor='#606060',markersize=5,label='Individual training seeds 0, 1, 2'),Line2D([],[],linestyle='',marker='o',color='#333333',markersize=6,label='Mean ± sample SD across training seeds'),Line2D([],[],linestyle=(0,(4,3)),color='#AAAAAA',linewidth=.8,label='Matched Uniform = 1')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.04),ncol=3,frameon=False,fontsize=8.8,handletextpad=.55,columnspacing=1.7)
    name='standard_training_seed_rauec'
    fig.savefig(args.output_dir/(name+'.pdf'),metadata={'Title':'Standard benchmark training-seed rAUEC','Creator':'Matplotlib','CreationDate':None,'ModDate':None})
    fig.savefig(args.output_dir/(name+'.svg'),metadata={'Title':'Standard benchmark training-seed rAUEC','Creator':'Matplotlib','Date':None})
    fig.savefig(args.output_dir/(name+'.png'),dpi=300,metadata={'Software':'Matplotlib','Title':'Standard benchmark training-seed rAUEC'})
    plt.close(fig)
    source_path=args.output_dir/(name+'_source.csv');source_path.write_bytes(args.data.read_bytes())
    caption='Field rAUEC over three independently trained Transolver3 models on each standard benchmark. Open markers show training seeds 0, 1 and 2 from left to right; filled markers and bars show their mean and sample standard deviation. Within each training run, the three-budget trapezoidal area ratio is computed separately for sampling seeds 0, 1 and 2 using matched Uniform denominators, then averaged. Full 100% is excluded from the integral. Every training run uses the same 200 test cases. Darcy uses the historical cohort with 1000 training and 200 validation cases, which differs from the matched LinearNO cohort with a 900/100 split. Pipe and Darcy retain their documented history of test-informed policy development.\n'
    (args.output_dir/(name+'.caption.md')).write_text(caption)
    manifest=dict(source_csv=args.data.name,source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),backbone='Transolver3',training_seeds=[0,1,2],sampling_seeds_per_training_run=[0,1,2],uncertainty='sample SD across3independent training runs, ddof1',formats=['PDF','SVG','PNG300dpi'],palette_origin='existing project figure_tools/style.py',method_colors=COLORS,figure_type='point estimates and error bars with individual runs; common y-axis; no bars from a truncated baseline',files={str(p.name):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.output_dir.glob(name+'*') if p.is_file()})
    (args.output_dir/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figure':name,'datasets':3,'CSV_rows':12,'outputs':str(args.output_dir)}))


if __name__=='__main__':main()
