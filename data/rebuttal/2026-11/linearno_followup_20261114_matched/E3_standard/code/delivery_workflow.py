"""Keep accepted preliminary/final standard results visible while jobs run."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from collect_results import BASE, FOLLOWUP, read, write_json


def receipt_signature():
    files = []
    for dataset in ['darcy', 'airfoil']:
        folder = BASE / dataset
        for pattern in ['results/*completion.json', 'results/*acceptance.json',
                        'records/*/*completion.json', 'results/backbone_rauec.csv']:
            files.extend(folder.glob(pattern))
    return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files)


def report():
    state = read(BASE / 'results/collection_status.json')
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    lines = ['# LinearNO standard benchmark follow-up', '', f'Generated from current receipts at {now}.', '',
             'The main comparison uses Transolver3 and the official LinearNO point-temp model on Darcy 85×85 and Airfoil 221×51. '
             'The official convolution variant is a separate Full-only reference. Both point models share a new 900/100 training/validation split, '
             'training seed 0, and the same task-specific point schedule. Historical test visibility is disclosed in protocol.md.', '',
             '| Dataset | Backbone | Accuracy stage | Field rows | Reason if pending |', '| --- | --- | --- | ---: | --- |']
    for row in state.get('models', []):
        lines.append(f"| {row['dataset']} | {row['backbone']} | {row['accuracy_status']} | {row.get('actual_field_rows', 'N/A')} | {row['reason']} |")
    lines += ['', f"Complete accuracy matrix: {state.get('accuracy_complete', False)}. Complete A/B/C latency matrix: {state.get('latency_complete', False)}.", '',
              f"Current independently checked field rows: {state.get('field_rows', 0)}; rAUEC rows: {state.get('rauec_rows', 0)}; formal timing conditions: {state.get('latency_conditions', 0)} / 2,646.", '',
              'The independent collector verifies the predeclared test IDs, checkpoint and normalizer, exact method/budget/seed matrix, and CPG beta pairing. '
              'rAUEC integrates each seed’s case-average error curve at 37.5%, 50%, and 75%, divides by its own matched Uniform area, then reports mean and sample SD. '
              'Full 100% does not enter the integral. Each backbone has its own denominator.', '',
              'Final predictions use reconstruction v1.1: original QHull linear interpolation with nearest fallback, and a sparse application of the same weights. '
              'Both reconstructions consume the same fixed predictions. Each new checkpoint must pass validation and full-test equivalence checks.', '',
              'Tables and figures are generated from results/*.csv by code/render_results.py. The render manifest records source CSVs; visual QA is separate. '
              'Missing cohorts or CA rows are N/A, and incomplete timing is not replaced with a single evaluation-call duration. '
              'Full-only convolution results are kept out of rAUEC.', '',
              'Primary timing A is CPU input to a complete float64 CPU field. B is GPU-resident input to a complete float64 GPU field. '
              'C reuses only an interpolation map for identical geometry and retained point order. '
              'Formal results contain 20 warmups and 100 direct wall-time repeats per condition; reconstruction profiling and memory runs are separate.', '',
              'Training and raw arrays remain indexed in the task directories. The local archive excludes checkpoint payloads and large training caches. '
              'Experiment scripts perform no Git writes. The user separately authorized root to publish the reviewed current snapshot to an independent branch; '
              'see ../coordination/publication.json. Main and Feishu remain unchanged.', '']
    (BASE / 'REPORT.md').write_text('\n'.join(lines))
    claims = ['# Evidence and claim card', '',
              '| Question | Current support |', '| --- | --- |',
              '| Does the official point-temp accept variable point sets? | Actual-class interface checks in both task results directories. |',
              '| Does official conv-temp accept arbitrary retained point sets? | No; native fixed-grid reshape is incompatible. Its Full reference is trained separately. |',
              '| Does reconstruction preserve the mathematical rule? | Shared fixed-prediction validation passed; each newly trained checkpoint has a separate required acceptance receipt. |',
              f"| Are all new-task final accuracy rows present? | {state.get('accuracy_complete', False)}; see model states in REPORT.md. |",
              f"| Are all formal cost conditions complete? | {state.get('latency_complete', False)}; {state.get('latency_conditions', 0)}/2646 conditions collected. |",
              '| Is beta=1 always best? | No universal guarantee is claimed. Use the paired beta=0/1 table with identical point sets. |',
              '| Which backbone is more accurate? | Compare complete physical field errors, including each same-checkpoint Full reference. rAUEC values use different backbone denominators. |',
              '| Is optimized reconstruction faster? | Only accepted same-scope old/new direct timing supports this; CPU diagnostics are not a GPU speed result. Slowdowns are retained. |',
              '| Is this an untouched test population? | No; the historical populations were previously viewed. New training selection uses the locked validation split. |',
              '| Were results published? | Independent-branch publication is authorized; the remote commit and exact scope are recorded in ../coordination/publication.json. |', '']
    (BASE / 'CLAIM_CARD.md').write_text('\n'.join(claims))
    (BASE / 'read.me').write_text(f"Standard follow-up updated {now}. Accuracy complete={state.get('accuracy_complete', False)}, latency complete={state.get('latency_complete', False)}.\n"
        'Live epochs and resource waits: ../read.me. Independent statistical state: results/collection_status.json.\n'
        'REPORT.md and CLAIM_CARD.md distinguish interface checks, preliminary accuracy, final accuracy, formal latency and pending visual QA.\n'
        'Experiment workers perform no remote writes; root branch publication is recorded in ../coordination/publication.json.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    directory = FOLLOWUP / 'coordination'
    signature = None
    with (directory / 'delivery_workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            current = receipt_signature()
            if current != signature:
                state = dict(pid=os.getpid(), execution_status='running', updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                             source_signature=current, stages={})
                for script in ['collect_results.py', 'render_results.py']:
                    command = [sys.executable, str(Path(__file__).with_name(script))]
                    process = subprocess.run(command, cwd=FOLLOWUP, capture_output=True, text=True)
                    state['stages'][script] = dict(command=command, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
                    if process.returncode:
                        state['execution_status'] = 'failed'
                        break
                else:
                    report()
                    state['execution_status'] = 'waiting_for_new_receipt'
                    signature = current
                write_json(directory / 'delivery_workflow.json', state)
            if not args.watch:
                break
            time.sleep(45)


if __name__ == '__main__':
    main()
