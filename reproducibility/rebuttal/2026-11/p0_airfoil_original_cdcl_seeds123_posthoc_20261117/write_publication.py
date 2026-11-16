#!/usr/bin/env python3
"""Prepare truthful, explicitly post-hoc publication text after numeric verification."""
from __future__ import annotations
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent/'p0_airfoil_original_cdcl_20261117'
REPO=ORIGINAL/'paper'
NAME='p0_airfoil_original_cdcl_seeds123_posthoc_20261117'
DATA=REPO/'data/rebuttal/2026-11'/NAME
REPRO=REPO/'reproducibility/rebuttal/2026-11'/NAME
PARENT=REPO/'data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_csv(p):return list(csv.DictReader(Path(p).open()))
def write_json(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def after_title(path,block):
    text=path.read_text();marker='<!-- airfoil-seeds123-posthoc -->'
    if marker in text:raise RuntimeError(f'already annotated: {path}')
    title,_,rest=text.partition('\n')
    path.write_text(title+'\n\n'+marker+'\n'+block+'\n<!-- /airfoil-seeds123-posthoc -->\n'+rest)


def main():
    verification=json.loads((DATA/'verification.json').read_text())
    public=json.loads((ROOT/'public_numeric_verification.json').read_text())
    assert verification['complete'] and public['passed']
    assert verification['analysis_design']=='post_hoc_exploratory_subset'
    assert verification['display_training_seeds']==[1,2,3]
    verification['public_csv_recomputation']=public
    write_json(DATA/'verification.json',verification)
    shutil.copy2(ROOT/'public_numeric_verification.json',DATA/'public_numeric_verification.json')
    rows=[r for r in read_csv(DATA/'training_seed_cdcl_summary.csv') if r['method']!='Uniform']
    ca_row=next(r for r in rows if r['method']=='CA')
    ca_note=(f"CA remains above the Uniform reference in this display: its subset mean is {float(ca_row['mean']):.6f}, "
             f"and the seed3 value is {float(ca_row['training_seed_3']):.6f}. "
             "For case1198, seed3 CA at budget0.50 and sampling seed2 predicts Cl=0.0002118407 and has Cd/Cl absolute error92.266864. "
             "This observation is retained and is not classified as an invalid training run.")
    table='| Method | Training seed 1 | Training seed 2 | Training seed 3 | Subset mean ± sample SD |\n|---|---:|---:|---:|---:|\n'
    for row in rows:
        table+='| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} ± {:.6f} |\n'.format(row['method'],*[float(row[f'training_seed_{s}']) for s in [1,2,3]],float(row['mean']),float(row['sample_sd']))
    detail='| Training seed | Method | Sampling seed 0 | Sampling seed 1 | Sampling seed 2 | Sampling mean ± SD |\n|---:|---|---:|---:|---:|---:|\n'
    for row in read_csv(DATA/'per_training_seed_summary.csv'):
        if row['method']=='Uniform':continue
        detail+='| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} ± {:.6f} |\n'.format(row['training_seed'],row['method'],*[float(row[f'sampling_seed_{s}']) for s in range(3)],float(row['rAUEC']),float(row['sampling_sd']))
    note=('**Post-hoc exploratory subset.** Training seed 0 was excluded from this display after its test results were observed. '
          'It passed the original execution and protocol checks; the exclusion is not an invalid-run correction. '
          'The complete original seeds 0/1/2 data and summaries remain in the [original corrected package](../p0_airfoil_original_cdcl_20261117/README.md). '
          'The displayed seeds 1/2/3 therefore cannot be presented as an unselected three-seed stability validation.')
    readme='# Airfoil Cd/Cl: exploratory display of training seeds 1/2/3\n\n'+note+'\n\n'
    readme+='Seed 3 was fixed before its training and test evaluation. It uses the same P0 Transolver3 architecture, 900/100 split, normalizer, 1,000 training epochs and 113,000 updates. Its own pressure head was trained for 200 epochs. No further seed was selected according to seed 3 performance.\n\n'
    readme+='The table uses original Cd/Cl-specific policies, all 110 original cases, sampling seeds 0/1/2, and budgets 37.5%, 50%, 75%. CPG uses 10% coarse sampling and CA retains the original 5%. Original reconstruction and pressure-force integration are unchanged. Uniform rAUEC is1.\n\n'+table
    readme+='\n'+ca_note+'\n'
    readme+='\nThese SDs describe the three displayed checkpoints. They are not estimates from an unselected training-seed sample. Each checkpoint value first averages three sampling-seed rAUECs. Each sampling ratio uses its own checkpoint/sampling-seed Uniform area after case averaging and trapezoidal integration. Full 100% is excluded.\n\n'
    readme+='Seed1/2 contribute 7,920 records reused from their already verified **original-Cd/Cl** evaluations, and seed 3 contributes 3,960 fresh records. No P0 field-policy force diagnostics are used. The separate original frozen-checkpoint control is reused with its exact3,960-record reproduction evidence. All110 cases, including1007/1059/1198, remain; exclusions1041/1139 follow the original Cl>0.02 protocol.\n\n'
    readme+='Files: [per-case records](per_case_records.csv.gz), [new seed 3 records](seed3_per_case_records.csv.gz), [per-budget/sampling table](per_seed_sampling_summary.csv), [per-checkpoint sampling SD](per_training_seed_summary.csv), [training subset table](training_seed_cdcl_summary.csv), [Uniform denominators](uniform_denominators.csv), [checkpoint/head registry](checkpoint_registry.csv), [case diagnostics](case_1007_1059_1198.csv), [verification](verification.json), [protocol](protocol_original_cdcl.json), [source hashes](source_hashes.json), [reproduction](reproduction_command.md), and [English response](reviewer_reply.md).\n\n'
    readme+='The completed scope is Transolver3 original-Cd/Cl accuracy for this disclosed exploratory subset. This package makes no completion claim for Plasticity three-training-seed experiments or new formal latency of the Cd/Cl policies.\n'
    (DATA/'README.md').write_text(readme)
    (DATA/'RESULTS_DETAIL.md').write_text('# Exploratory seeds 1/2/3 numerical results\n\n'+note+'\n\n'+table+'\nWithin-checkpoint sampling variation is separate:\n\n'+detail)
    original=read_csv(PARENT/'training_seed_cdcl_summary.csv')
    original_values='; '.join(f"{r['method']} {float(r['mean']):.6f} ± {float(r['sample_sd']):.6f}" for r in original if r['method']!='Uniform')
    reply=('The original Table 1 error bars represent three inference-sampling seeds of one frozen checkpoint. '
           'We subsequently evaluated independent Transolver3 training seeds 0/1/2 under the original Cd/Cl policies. '
           'After observing those results, we examined an exploratory display excluding training seed 0 and adding a fixed new training seed 3. '
           'The excluded run passed execution and protocol checks. Its data remain available, and this exclusion is explicitly post hoc.\n\n'
           'Seed 3 uses the same 900/100 training/validation protocol, normalizer, 1,000 epochs and its own CA head trained for 200 epochs. '
           'The evaluation retains all 110 original cases, original Cd/Cl-specific PI/CPG/CA policies, three sampling seeds, three budgets, '
           'original reconstruction and force integration. The original frozen-checkpoint control reproduced all 3,960 published case errors exactly.\n\n'+table+
           '\n'+ca_note+'\n\nThe reported SD is across the three displayed training checkpoints. Sampling SD within each checkpoint is reported separately. '
           'We first average case errors at each budget, integrate the curve and divide by the Uniform area for the same checkpoint and sampling seed, '
           'then average sampling ratios within the checkpoint. Full 100% is outside the integral.\n\n'
           'Because the displayed subset was chosen after the initial results were known, these values cannot establish unselected training-seed robustness. '
           'The original complete seeds 0/1/2 results remain '+original_values+'. '
           'Airfoil Cd/Cl is sensitive to small predicted lift, which affects both method errors and matched Uniform denominators. '
           'All required cases 1007/1059/1198 and all large errors are retained. We do not calculate paired improvement deltas between different checkpoints. '
           'Plasticity is outside this three-checkpoint supplement.\n\n'
           'Supporting data: [exploratory original-Cd/Cl package](https://github.com/yl602019618/AdaSolver_Paper/tree/rebuttal/20261114-linearno-seeds/data/rebuttal/2026-11/'+NAME+').\n')
    (DATA/'reviewer_reply.md').write_text(reply)
    (REPO/'reviews/airfoil-seeds123-posthoc-response.md').write_text(reply)
    REPRO.mkdir(parents=True,exist_ok=True)
    for name in ['verify_posthoc.py','finalize_results.py','write_publication.py','refresh_manifests.py','run_pipeline.py']:
        shutil.copy2(ROOT/name,REPRO/name)
    for name in ['protocol.json','preflight.json','launch.json','pipeline_status.json']:
        shutil.copy2(ROOT/name,DATA/('extension_'+name))
    shutil.copytree(ROOT/'run/code',REPRO/'training_source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'run/config',REPRO/'training_config',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'run/checkpoints/t3/history.jsonl',DATA/'seed3_training_history.jsonl')
    shutil.copy2(ROOT/'run/checkpoints/t3/manifest.json',DATA/'seed3_backbone_manifest.json')
    shutil.copy2(ROOT/'run/checkpoints/t3/head/manifest.json',DATA/'seed3_head_manifest.json')
    shutil.copy2(ROOT/'run/results/t3_head_acceptance.json',DATA/'seed3_head_acceptance.json')
    (DATA/'reproduction_command.md').write_text('# Verification and reproduction\n\nFrom the repository root, recompute the public records and subset statistics:\n\n```sh\npython3 reproducibility/rebuttal/2026-11/'+NAME+'/verify_posthoc.py --data-dir data/rebuttal/2026-11/'+NAME+'\n```\n\nThe actual fixed-seed protocol, launch and pipeline receipts are included as extension_*.json. Training source/configs are under the accompanying reproduction directory. They use Python3.10.12, PyTorch2.10.0+cu128, CUDA12.8, the same P0 cache/normalizer and external checkpoints identified in the registry. The full native seed 3 commands are in extension_pipeline_status.json. The original Cd/Cl evaluator and reference files remain in reproducibility/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/. Dataset and checkpoint path bindings refer to the recorded workspace. No seed 0 data are deleted or relabeled.\n')
    (REPRO/'README.md').write_text('# Exploratory seeds 1/2/3 reproduction\n\nSee [reproduction commands](../../../../data/rebuttal/2026-11/'+NAME+'/reproduction_command.md). This is a post-hoc subset; verification establishes arithmetic and record completeness, not unbiased seed selection.\n')
    link='data/rebuttal/2026-11/'+NAME
    root_note=('**Exploratory display: training seeds 1/2/3.** Seed 0 was excluded after its test performance was observed, while its valid original data remain preserved. '
               '[Subset results]('+link+'/README.md) · [Displayed values](RESULTS_DETAIL.md) · [English response](reviews/airfoil-seeds123-posthoc-response.md). '
               'This selected display is not an unselected three-seed stability validation.')
    after_title(REPO/'README.md',root_note)
    after_title(REPO/'TESTED_CONTENTS.md',root_note)
    after_title(REPO/'rebuttal/2026-11/RESULTS_INDEX.md',root_note.replace('](','](../../'))
    (REPO/'RESULTS_DETAIL.md').write_text('# Current exploratory Airfoil Cd/Cl display\n\n'+root_note+'\n\n'+table+'\n[Complete original seeds 0/1/2 evidence](data/rebuttal/2026-11/p0_airfoil_original_cdcl_20261117/README.md) remains preserved.\n')
    readme=REPO/'read.me'
    previous=readme.read_text()
    readme.write_text('2026-11-17 — Post-hoc display of training seeds 1/2/3\n\nCurrent display package: '+link+'\nSeed0 is excluded from this exploratory display after its test results were observed.\nThe run was valid; original012 records and statistics remain preserved.\nSeed3 was fixed before training and was run once for1000 epochs with its own200-epoch head.\nDisplay data:11880 records (7920 reused original-Cd/Cl records for1/2;3960 new seed 3 records).\nRecord verification and public CSV recomputation passed. Selection remains post hoc.\nNo unselected training-seed robustness claim and no Plasticity completion claim.\n\nEarlier original012 release entry follows:\n\n'+previous)
    ledger=REPO/'reviews/revision-ledger.md'
    with ledger.open('a') as f:
        f.write('| Exploratory Airfoil seeds 1/2/3 display | Seed 0 exclusion requested after observing test performance. | Preserve original012 evidence; train fixed seed 3 once and disclose post-hoc subset in every active result index and response. | [Subset](../'+link+'/README.md); [verification](../'+link+'/verification.json). | Done as an exploratory display; no unselected robustness claim. |\n')
    after_title(PARENT/'README.md','A later [exploratory seeds 1/2/3 display](../'+NAME+'/README.md) excludes the valid P0 seed 0 after observing its result. The complete original012 records and statistics in this package remain unchanged.')
    print('Publication documents prepared with explicit post-hoc exclusion and original results preserved.')


if __name__=='__main__':main()
