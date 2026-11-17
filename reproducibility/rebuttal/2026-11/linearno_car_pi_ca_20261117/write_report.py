"""Write the completed PI/CA supplement without changing other experiments."""
import csv
import json
from pathlib import Path
import shutil
import statistics
from common import ROOT,sha256,write_json

NAME='linearno_car_pi_ca_20261117'
PAPER=ROOT/'paper'
DATA=PAPER/'data/rebuttal/2026-11'/NAME
REPRO=PAPER/'reproducibility/rebuttal/2026-11'/NAME


def rows(p):return list(csv.DictReader(Path(p).open()))


def insert(path,text):
    marker='<!-- linearno-car-pi-ca-20261117 -->'
    old=path.read_text();assert marker not in old,path
    title,_,rest=old.partition('\n');path.write_text(title+'\n\n'+marker+'\n'+text+'\n<!-- /linearno-car-pi-ca-20261117 -->\n'+rest)


def main():
    verified=json.loads((DATA/'verification.json').read_text());public=json.loads((ROOT/'public_numeric_verification.json').read_text())
    assert verified['complete'] and public['passed'];verified['public_numeric_recomputation']=public;write_json(DATA/'verification.json',verified)
    shutil.copy2(ROOT/'public_numeric_verification.json',DATA/'public_numeric_verification.json')
    accuracy={(r['target'],r['method']):r for r in rows(DATA/'accuracy_summary.csv')}
    table='| Method | Field rAUEC ↓ | Cd rAUEC ↓ |\n|---|---:|---:|\n'
    for method,display in [('uniform','Uniform'),('pi','PI'),('cpg','CPG (β=1)'),('ca','CA')]:
        a,b=accuracy['field',method],accuracy['cd',method]
        table+=f"| {display} | {float(a['mean']):.6f} ± {float(a['sampling_sd']):.6f} | {float(b['mean']):.6f} ± {float(b['sampling_sd']):.6f} |\n"
    timing=rows(DATA/'inference_budget_summary.csv')
    lat='| Policy target | Method | Retained points | Complete CPU-field request / ms | Full/request latency ratio |\n|---|---|---:|---:|---:|\n'
    for target in ['field','cd']:
        for method in ['fullmesh','uniform','pi','cpg','ca']:
            rr=next(r for r in timing if r['target']==target and r['method']==method and float(r['budget'])==(1. if method=='fullmesh' else .375))
            lat+=f"| {target.title()} | {'Full' if method=='fullmesh' else method.upper()} | {100 if method=='fullmesh' else 37.5}% | {float(rr['mean_condition_median_ms']):.3f} | {float(rr['full_over_method_latency_ratio']):.3f} |\n"
    compared=[r for r in timing if r['method'] in ['pi','ca']]
    slower=sum(float(r['full_over_method_latency_ratio'])<1 for r in compared)
    cost=f'Across the {len(compared)} PI/CA target–budget summaries, {slower} have higher complete-request latency than their same-GPU Full reference. These measurements do not turn reduced point counts into an unconditional end-to-end speedup claim.'
    head=json.loads((DATA/'head_training_manifest.json').read_text())
    readme='''# ShapeNet-Car LinearNO: PI and CA completion

Status: **VERIFIED_COMPLETE** for the scope below. This supplement adds the missing PI/CA training and inference path to the existing official ShapeNet-Car LinearNO training-seed0 checkpoint.

The backbone is the complete official eight-block LinearNO, with hidden width256, eight heads, seven native input channels, four output channels and3,852,420 parameters. Its existing400-epoch/142,000-update checkpoint and normalizer are reused. PI is geometry-only and requires no fitted head. CA uses a new256–64–1 scalar pressure head trained on this LinearNO's own first-block features at exactly10% points, with710 training cases,39 validation cases, two masks per case and200 epochs. It never loads a Transolver head or reuses a Transolver slice cache. Final CA prediction executes the complete official LinearNO with point-key compensation.

All100 original fold0 test cases, three point budgets (37.5%,50%,75%) and sampling seeds0/1/2 are evaluated. Field and Cd use their separate frozen metric-specific policies. Field error is velocity relative L2 plus0.5 times surface-pressure relative L2; Cd uses relative absolute total-drag error. In each sampling run, case means are integrated across the three budgets and normalized by the matching Uniform area from the same checkpoint. Full100% is a separate reference and is not integrated.

'''+table+'''
The ± values are **sampling-seed sample SDs** for one trained backbone. No independent-training-seed SD is claimed. Existing Full/Uniform/CPG controls reproduce all4,600 native rows exactly; new PI/CA policies add3,600 rows, giving8,200 native requests and9,200 primary metric views. Cross-target diagnostic metrics remain in the raw records but do not replace metric-specific Cd or Field results. CPG β0/1 accuracy controls, including its small field degradation at β1, are retained separately.

The pressure head's1,498 real coarse mask groups and550,561 stored surface rows passed acceptance. Every target and selected index was checked, its saved validation MSE was reproduced, and its first-block feature output matched the official model's hook exactly on a real geometry.

Complete-output timings at37.5% (plus the100% Full reference) are:

'''+lat+'\n'+cost+'''

The timed call starts from CPU-resident normalized inputs and a mesh-derived pressure-influence vector and returns a complete physical float64 CPU field. Method-specific geometry/scorer setup, dynamic acquisition, selection, compensation, input transfer, native final forward, decoding, k4 reconstruction and output transfer are included. Model/data loading, mesh-influence preparation and offline force/error evaluation are outside this request. Sampling and predictions are recomputed for every call; no field prediction or selected-set cache supplies the timed output.

Timing covers three fixed E3 geometries, three sampling seeds and all three budgets. Each condition has20 warmups and100 formal measurements. Field and Cd policies were measured on separate exclusive H20 GPUs,4 and5, respectively, with their own Full/Uniform/CPG references. The table reports means of nine case/sampling condition medians (three case medians for Full), not a pooled percentile. All222 conditions,22,200 formal repetitions and4,440 warmups are included. Before each timing condition, output metrics and exact selected/coarse indices are checked against native accuracy records. Per-condition p95 and every repetition are available below. No isolated per-method memory result is claimed.

| Artifact | Contents |
|---|---|
| [per_case_records.csv.gz](per_case_records.csv.gz) | All8,200 native requests, both measured metrics and policy/checkpoint/point identities |
| [primary_metric_records.csv.gz](primary_metric_records.csv.gz) |9,200 explicitly assigned metric-specific views |
| [accuracy_summary.csv](accuracy_summary.csv) | Per-sampling-seed rAUEC, mean and sampling SD |
| [per_sampling_seed_budget.csv](per_sampling_seed_budget.csv) | Case-mean curves, method/Uniform areas and rAUEC |
| [uniform_denominators.csv](uniform_denominators.csv) | Six target/sampling-seed denominators |
| [frozen_policies.json](frozen_policies.json) | Frozen Field/Cd configurations |
| [checkpoint_registry.csv](checkpoint_registry.csv) | Backbone/head identities and external backbone locations |
| [head_inference_manifest.json](head_inference_manifest.json) | Portable pointer to the included trained head weights |
| [head_training_history.csv](head_training_history.csv) | Complete200-epoch head history |
| [inference_budget_summary.csv](inference_budget_summary.csv) | All budgets and explicit same-GPU comparisons |
| [inference_conditions.csv](inference_conditions.csv) |222 condition median/p95/mean/repeat-SD measurements |
| [inference_repeats.csv.gz](inference_repeats.csv.gz) | Every formal and warm-up wall-clock observation |
| [verification.json](verification.json) | Head, control, coverage, policy and independent numeric checks |
| [reproduction_command.md](reproduction_command.md) | Exact workspace execution and public CSV-only verification |

This completion is for training seed0. Independent LinearNO Car backbone seeds1/2, B/C reconstruction scenarios and isolated per-method memory remain outside this package. The official model implementation is referenced at its pinned upstream commit and is not bundled; the authored adapters, training/evaluation wrappers, inference callable and verification code are provided.
'''
    replacements={'width256':'width 256','and3,852,420':'and 3,852,420','existing400':'existing 400','new256':'new 256',
                  'exactly10%':'exactly 10%','with710':'with 710',',39':', 39','and200':'and 200','All100':'All 100',
                  'seeds0/1/2':'seeds 0/1/2','plus0.5':'plus 0.5','Full100%':'Full 100%',
                  'all4,600':'all 4,600','add3,600':'add 3,600','giving8,200':'giving 8,200','and9,200':'and 9,200',
                  "head's1,498":"head's 1,498",'and550,561':'and 550,561','at37.5%':'at 37.5%','the100%':'the 100%',
                  'has20':'has 20','and100':'and 100','GPUs,4':'GPUs, 4','All222':'All 222',',22,200':', 22,200','and4,440':'and 4,440',
                  'All8,200':'All 8,200','|9,200':'| 9,200','Complete200':'Complete 200','|222':'| 222','seed0':'seed 0','seeds1/2':'seeds 1/2'}
    for a,b in replacements.items():readme=readme.replace(a,b)
    (DATA/'README.md').write_text(readme)
    (DATA/'RESULTS_DETAIL.md').write_text('# LinearNO ShapeNet-Car PI/CA results\n\nOne backbone training seed0; mean±sample SD across three inference sampling seeds. Separate frozen policies for Field and Cd.\n\n'+table+'\nComplete CPU-field timing at37.5% follows; all budgets are in inference_budget_summary.csv.\n\n'+lat+'\n'+cost+'\n')
    reproduction='''# Reproduction and public verification

From the paper-repository root, validate the public numeric records and head artifact without a GPU:

```sh
python3 reproducibility/rebuttal/2026-11/linearno_car_pi_ca_20261117/verify_published.py --data-dir data/rebuttal/2026-11/linearno_car_pi_ca_20261117
```

Full execution uses the recorded workspace's E3 frozen source snapshot, official LinearNO checkpoint, E5 case split/normalizer and ShapeNet-Car data. Their paths and identities are in checkpoint_registry.csv and source_hashes.json. The official upstream is https://github.com/HiPRL/LinearNO at3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269; its full implementation is not redistributed here. Python3.10, PyTorch2.10.0+cu128 and CUDA12.8 were used; exact environment/device records are in the inference manifests.

Copy this reproduction directory to a fresh sibling of the original experiment directory, under `07_current/linearno_followup_20261114/reuse_review/<fresh_name>`. This retains the original relative workspace resolver. The frozen E3/E5 inputs and recorded external data must be present. Completed output folders are protected against overwrite.

From that fresh directory:

```sh
python3 bind_splits.py --manifest /mnt/ai-material-data/iclr2027_adatransolver/07_current/rebuttal_20261112/experiments/E3_backbones/checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json --output splits/seed0.json
python3 workflow.py --group head_adaptive --gpu 0
python3 workflow.py --group controls --gpu 3
python3 validate_trained_head.py --gpu 4
python3 verify_accuracy.py
python3 benchmark_inference.py --target field --gpu 4 --cpu-set 48,49,50,51
python3 benchmark_inference.py --target cd --gpu 5 --cpu-set 52,53,54,55
python3 package_results.py
```

Choose currently available GPUs/CPU cores and preserve their actual identities in the output. Head/adaptive and control workflows may run concurrently on separate GPUs. Each target's formal timing requires its exclusive GPU. The commands above preserve20 warmups and100 formal repetitions. The relevant current inference entry is `LinearNOCarRequest` in inference_request.py; older preparation helpers are retained solely to satisfy the frozen source lock and are not B/C timing evidence for this package.

The supplied `linearno_car_pressure_head_seed0.pt` and relative-path head_inference_manifest.json can be loaded with `LinearNOCA` alongside the exact backbone and normalizer in checkpoint_registry.csv. PI uses no trained head. The included backbone adapter/evaluation bridge references the external pinned native implementation.
'''
    (DATA/'reproduction_command.md').write_text(reproduction)
    shutil.copy2(ROOT/'verify_published.py',REPRO/'verify_published.py');shutil.copy2(ROOT/'write_report.py',REPRO/'write_report.py')
    shutil.copy2(ROOT/'refresh_release.py',REPRO/'refresh_release.py')
    (REPRO/'README.md').write_text('# LinearNO Car PI/CA reproduction\n\nSee [execution commands](../../../../data/rebuttal/2026-11/'+NAME+'/reproduction_command.md) and [verified results](../../../../data/rebuttal/2026-11/'+NAME+'/README.md). The authored bridges are included; the official native model, large backbone checkpoint and benchmark data are externally referenced.\n')
    rootpage='# LinearNO ShapeNet-Car PI/CA supplement\n\n'+table+'\nThe ± values are sampling SD for the existing trained seed0 backbone. One new matching CA head was trained for200epochs; PI requires no head.\n\n[Full package](data/rebuttal/2026-11/'+NAME+'/README.md) · [Verification](data/rebuttal/2026-11/'+NAME+'/verification.json) · [Inference costs](data/rebuttal/2026-11/'+NAME+'/inference_budget_summary.csv).\n'
    (PAPER/'LINEARNO_CAR_PI_CA.md').write_text(rootpage)
    message='**ShapeNet-Car LinearNO PI/CA complete (training seed0).** [Results and method scope](data/rebuttal/2026-11/'+NAME+'/README.md) · [Summary](LINEARNO_CAR_PI_CA.md). The native official backbone now has geometry-only PI and its own trained CA pressure head, separate Field/Cd policies,100-case accuracy and222 formal complete-CPU-field timing conditions. Error bars here measure sampling variability; independent training seeds1/2 are not included.'
    insert(PAPER/'README.md',message);insert(PAPER/'TESTED_CONTENTS.md',message)
    index=PAPER/'TESTED_CONTENTS.md';text=index.read_text()
    text=text.replace('Car LinearNO head 仍缺','本历史清单尚不含后来新增的 LinearNO head；seed0 PI/CA 最新完成情况见上方')
    text=text.replace('LinearNO PI/CA 与对应 head、training seeds1/2 仍缺','本历史 E3 快照仅含 Full/Uniform/CPG；seed0 PI/CA 与 head 已由新补充完成；独立 training seeds1/2 未纳入本包')
    text=text.replace('T3有PI/CA，LinearNO无','本历史 A/B/C 矩阵仅 T3 含 PI/CA；LinearNO 的新完整 CPU 输出组见上方')
    text=text.replace('LinearNO PI/CA/head与多training-seed结论仍未补齐','本历史 A/B/C 未含 LinearNO PI/CA；新独立 CPU 输出组已补齐；B/C 和多 training seed 不在本次范围')
    text=text.replace('Car LinearNO 的 PI/CA、对应 head 和独立 training seeds1/2 尚未完成','Car LinearNO seed0 的 PI/CA 及对应 head 已完成并单列于新补充；独立 training seeds1/2 未纳入本次完成范围')
    index.write_text(text)
    insert(PAPER/'rebuttal/2026-11/RESULTS_INDEX.md',message.replace('](','](../../'))
    insert(PAPER/'data/rebuttal/2026-11/E3/README.md','The missing **LinearNO seed0 PI/CA** component is now completed in the [PI/CA supplement](../'+NAME+'/README.md), including its own200-epoch head and separate Field/Cd policies. Existing E3 numeric files remain unchanged. The earlier CPG comparison below is preserved as its original scope.')
    (PAPER/'read.me').write_text('2026-11-17 — ShapeNet-Car LinearNO PI/CA completion\n\nPackage: data/rebuttal/2026-11/'+NAME+'\nScope: existing official training seed0 backbone, own200-epoch CA head; PI requires no extra training.\nAccuracy:100cases,3sampling seeds,3budgets;3600newPI/CA rows and4600exactcontrol reproductions.\nFormal inference:222conditions,22200formal repeats,4440warmups; fullCPUfloat64field output.\nStatus: VERIFIED_COMPLETE within this seed0scope; no independent-training-seed SD claim.\nOutside scope: backbone training seeds1/2, B/Cscenarios, isolatedmethodmemory.\n\nEarlier entries follow:\n\n'+(PAPER/'read.me').read_text())
    print('Written PI/CA result package, summary and branch entry points.')


if __name__=='__main__':main()
