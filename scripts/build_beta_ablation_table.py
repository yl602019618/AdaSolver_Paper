"""Build paired ablation comparisons from the final, full-precision results."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/beta_ablation'
ORDER = [('airfoil2d','field'),('pipe','field'),('darcy','field'),('elasticity','field'),
         ('plasticity2d','field'),('shapenet_car','field'),('driverml','field'),('superwing','field'),
         ('airfoil2d','cdcl'),('shapenet_car','cd'),('driverml','cd'),('superwing','cdcl')]
NAMES = {'airfoil2d':'Airfoil','pipe':'Pipe','darcy':'Darcy','elasticity':'Elasticity',
         'plasticity2d':'Plasticity','shapenet_car':'ShapeNet-Car','driverml':'DrivAerML','superwing':'SuperWing'}


def main():
    data = json.loads((DATA / 'summary.json').read_text())
    assert data['status']['complete']
    index = {(r['benchmark'],r['metric'],r['method'],r['beta']):r for r in data['results']}
    lines = [r'\begin{table}[!htbp]',r'\centering',r'\small',
             r'\caption{Measure correction on fixed adaptive samples. Entries are mean rAUEC over three sampling seeds, with Uniform equal to one. Bold marks the lower value in each within-policy $\beta=0$/$\beta^{\star}$ pair. Parentheses give the percentage decrease in mean rAUEC from $\beta=0$, computed before rounding. The compensation exponents $\beta^{\star}$ are specified in Appendix~\ref{app:density_policy_design}.}',
             r'\label{tab:beta_ablation}',r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',
             r'Dataset & Metric & \multicolumn{2}{c}{PI} & \multicolumn{2}{c}{CPG} & \multicolumn{2}{c}{CA} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'& & $\beta=0$ & $\beta^{\star}$ & $\beta=0$ & $\beta^{\star}$ & $\beta=0$ & $\beta^{\star}$ \\',
             r'\midrule']
    annotations = []
    for bench,metric in ORDER:
        cells=[]
        for method in ['pi','cpg','ca']:
            base=index[(bench,metric,method,0.)]
            selected=index[(bench,metric,method,base['production_beta'])]
            left,right=base['rauec_mean'],selected['rauec_mean']
            assert left>0
            decrease=100*(1-right/left)
            a,b=f'{left:.4f}',f'{right:.4f}'
            if left<=right:a=r'\textbf{'+a+'}'
            if right<=left:b=r'\textbf{'+b+'}'
            b+=r'\,{\footnotesize ('+f'{decrease:.1f}'+r'\%)}'
            cells.extend([a,b])
            annotations.append(dict(benchmark=bench,metric=metric,method=method,beta=selected['beta'],
                                    beta0_mean_rauec=left,selected_mean_rauec=right,
                                    mean_rauec_decrease_percent=decrease,
                                    bold_beta0=left<=right,bold_selected=right<=left))
        label={'field':'Field','cd':'Cd','cdcl':'Cd/Cl'}[metric]
        lines.append(NAMES[bench]+' & '+label+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'tables/beta_ablation.tex').write_text('\n'.join(lines)+'\n')
    with (DATA/'table_annotations.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(annotations[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(annotations)
    print('Built 36 within-policy comparisons from full-precision mean rAUECs.')


if __name__=='__main__':
    main()
