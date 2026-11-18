"""Run disjoint evaluation shards on a parent-owned GPU pair and accept them."""
from __future__ import annotations
import argparse,os,subprocess,sys,time
from common import OUT,CODE,read,write


def main(resolution):
    children=[];streams=[]
    receipt=OUT/f'results/r{resolution}_accuracy_acceptance.json'
    try:
        for shard in [0,1]:
            gate=OUT/f'evaluation/r{resolution}_all_shard{shard}of2_receipt.json'
            if gate.exists() and read(gate).get('status')=='complete':continue
            command=[sys.executable,str(CODE/'evaluate.py'),'--resolution',str(resolution),
                '--phase','all','--device',f'cuda:{shard}','--shard-count','2','--shard-index',str(shard)]
            stream=(OUT/f'logs/r{resolution}_evaluation_shard{shard}.log').open('a')
            streams.append(stream)
            children.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=OUT))
        while children and any(child.poll() is None for child in children):
            failed=[child for child in children if child.poll() not in [None,0]]
            if failed:raise RuntimeError(f'Evaluation shard failed with returncode {failed[0].returncode}')
            time.sleep(5)
        if any(child.returncode for child in children):raise RuntimeError('Evaluation shard failed')
        for shard in [0,1]:
            gate=read(OUT/f'evaluation/r{resolution}_all_shard{shard}of2_receipt.json')
            assert gate['passed'] and gate['status']=='complete' and gate['case_count']==100
            assert gate['request_count']==3700 and gate['metric_rows']==7400
        subprocess.run([sys.executable,str(CODE/'verify_results.py'),'--resolution',str(resolution)],cwd=OUT,check=True)
        assert read(receipt)['passed']
        print(receipt,flush=True)
    finally:
        # Only children started here are terminated on a local shard failure.
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            if child.poll() is None:child.wait(timeout=30)
        for stream in streams:stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,choices=[85,141,211],required=True)
    main(parser.parse_args().resolution)
