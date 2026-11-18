"""Isolated validation-only serialization and all-budget reconstruction check."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
import json,shutil
import numpy as np
import torch
from common import OUT as REAL_OUT,build_model,configure_precision,write
import evaluate,head_support


def main():
    configure_precision(0,4)
    fixture=REAL_OUT/'evaluation/engineering_record_fixture';fixture.mkdir(parents=True,exist_ok=True)
    if not (fixture/'data').exists():(fixture/'data').symlink_to(REAL_OUT/'data',target_is_directory=True)
    shutil.copyfile(REAL_OUT/'protocol.json',fixture/'protocol.json')
    (fixture/'README.md').write_text('# Engineering fixture only\n\nRandom initialized model/head, validation case 900 only. Values below are serialization and reconstruction checks, not experiment accuracy results. The data link is read-only in this check.\n')
    original_root=evaluate.OUT;original_model=evaluate.load_model;original_head=head_support.load_head
    scalar=evaluate.scalar_original();original_scalar=evaluate.scalar_original
    evaluate.OUT=fixture;evaluate.scalar_original=lambda:scalar
    def model(n,device):
        return build_model().to(device).eval().requires_grad_(False),{'checkpoint_sha256':'engineering-random-model'}
    def head(n,operator,device):
        return head_support.HeadAcquisition(operator,head_support.ScoreHead(),{'target_mean':1.,'target_std':1.}).to(device).eval(),{'checkpoint_sha256':'engineering-random-head'}
    evaluate.load_model=model;head_support.load_head=head
    checked=[]
    try:
        for n in [85,141,211]:
            runtime=evaluate.Runtime(n,'cpu',True)
            try:
                records=[]
                for method in evaluate.METHODS:
                    for budget in ([1.] if method=='full' else evaluate.BUDGETS):
                        row=runtime.request('validation',900,method,budget,None if method=='full' else 0,'engineering',validate=True)
                        assert row['validation_parity']['passed'];records.append(row)
                write(fixture/f'r{n}_validation_serialization.json',dict(engineering_fixture_only=True,trained_weights=False,records=records))
                checked.append(dict(resolution=n,requests=len(records),json_serialization_passed=True,all_budgets_checked=True))
                print(json.dumps(checked[-1]),flush=True)
            finally:runtime.close()
    finally:
        evaluate.OUT=original_root;evaluate.load_model=original_model;head_support.load_head=original_head;evaluate.scalar_original=original_scalar
    write(REAL_OUT/'evaluation/record_serialization_preflight.json',dict(passed=True,validation_ids=[900],test_opened=False,
        engineering_fixture_only=True,trained_weights=False,records=checked,fixture_directory=str(fixture)))


if __name__=='__main__':main()
