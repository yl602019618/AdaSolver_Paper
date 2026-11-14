"""Generate the README's metric table directly from its source CSVs."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    areas = list(csv.DictReader((ROOT / 'E3_standard/results/backbone_rauec.csv').open()))
    errors = list(csv.DictReader((ROOT / 'E3_standard/results/backbone_absolute_errors.csv').open()))
    protocol = 'grid_qhull_sparse_apply_v1p1'
    backbones = ['Transolver3', 'LinearNO_point_temp']
    lines = ['| 指标 / 方法 | Transolver3 | 官方 LinearNO point-temp |', '| --- | ---: | ---: |']
    full = [next(r for r in errors if r['backbone']==b and r['method']=='full' and r['reconstruction_id']==protocol) for b in backbones]
    lines.append('| Full：physical relative L2 | ' + ' | '.join(f"{float(r['mean_error']):.10f}" for r in full) + ' |')
    for method, label in [('uniform','Uniform'), ('pi','PI'), ('cpg_beta0','CPG β=0'), ('cpg','CPG β=1'), ('ca','CA')]:
        selected = [next(r for r in areas if r['backbone']==b and r['method']==method and r['reconstruction_id']==protocol) for b in backbones]
        lines.append('| ' + label + ' rAUEC | ' + ' | '.join(f"{float(r['mean']):.6f} ± {float(r['sampling_sd']):.6f}" for r in selected) + ' |')
    path = ROOT / 'README.md'; old = path.read_text()
    start = old.index('| 指标 / 方法 |'); end = old.index('\n\n', start)
    path.write_text(old[:start] + '\n'.join(lines) + old[end:])
    conv = next(r for r in errors if r['backbone']=='LinearNO_conv_temp' and r['method']=='full' and r['reconstruction_id']==protocol)
    text = path.read_text()
    text = text.replace('CONV_FULL_ERROR_FROM_CSV', f"{float(conv['mean_error']):.10f}")
    path.write_text(text)


if __name__ == '__main__':
    main()
