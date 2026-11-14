"""Each backbone trains its own complete pressure head on its own block0 features."""
from common import *
import argparse,time,subprocess
from adasolver.inference.airfoil2d_contract import coarse_count
from adasolver.inference.uniform.airfoil2d import sample_airfoil_uniform
from adasolver.train.airfoil2d_pressure_head import train as train_pressure_head

TRAIN_HEAD_SOURCE_BYTES=Path(__file__).read_bytes()
TRAIN_HEAD_SOURCE_SHA256=hashlib.sha256(TRAIN_HEAD_SOURCE_BYTES).hexdigest()


@torch.no_grad()
def build_bank(architecture):
    model,operator=load_model(architecture);data=Data();root=OUT/'cache'/f'{architecture}_head';root.mkdir(exist_ok=True)
    manifest_path=root/'manifest.json'
    if manifest_path.exists() and read(manifest_path).get('status')=='complete':
        existing=read(manifest_path)
        assert existing['operator_sha256']==operator['checkpoint_sha256']
        assert existing['normalizer_sha256']==operator['normalizer_sha256']
        assert existing['train_ids']==data.splits['training_ids'] and existing['validation_ids']==data.splits['validation_ids']
        return root
    mandatory=np.arange(50,171,dtype=np.int64)*51;shards=[];start=time.perf_counter()
    for split,ids in [('train',data.splits['training_ids']),('validation',data.splits['validation_ids'])]:
        for fraction in [.05,.1]:
            count=coarse_count(11271,fraction,121);name=f'{split}_cf{fraction:g}'
            number=len(ids)*2
            features=np.lib.format.open_memmap(root/f'{name}_features.npy',mode='w+',dtype=np.float32,shape=(number,count,128))
            targets=np.lib.format.open_memmap(root/f'{name}_target.npy',mode='w+',dtype=np.float32,shape=(number,count,1))
            indices=np.lib.format.open_memmap(root/f'{name}_indices.npy',mode='w+',dtype=np.int64,shape=(number,count))
            entries=[]
            for local,sid in enumerate(ids):
                coords,pressure=data.sample(sid)
                for mask_id in range(2):
                    seed=(10_000_000 if split=='train' else 20_000_000)+1000*sid+mask_id
                    chosen=sample_airfoil_uniform(11271,count,mandatory,np.random.default_rng(seed)).indices
                    pos=torch.as_tensor(coords[chosen],device='cuda:0')[None]
                    feature=feature0(model,pos)[0].cpu().numpy();row=local*2+mask_id
                    features[row]=feature;targets[row]=(pressure[chosen]-data.mean)/data.std;indices[row]=chosen
                    entries.append(dict(sample_id=sid,mask_seed=seed))
                if (local+1)%100==0:print(architecture,name,local+1,'/',len(ids),flush=True)
            for array in [features,targets,indices]:array.flush()
            shards.append(dict(name=name,split=split,fraction=fraction,coarse_count=count,records=number,entries=entries,
                               features=f'{name}_features.npy',target=f'{name}_target.npy',indices=f'{name}_indices.npy'))
    manifest=dict(benchmark='airfoil2d',version='followup20261114',status='complete',
        backbone=architecture,operator_checkpoint=operator['checkpoint'],operator_manifest=str(OUT/f'checkpoints/{architecture}/manifest.json'),
        operator_sha256=operator['checkpoint_sha256'],normalizer_sha256=operator['normalizer_sha256'],
        target='zscore_pressure',target_mean=float(data.mean),target_std=float(data.std),feature_tap='frozen_block0_on_S0',
        hidden_dim=128,nfull=11271,train_ids=data.splits['training_ids'],validation_ids=data.splits['validation_ids'],
        masks_per_geometry_fraction=2,fractions=[.05,.1],coarse_cap_count=1127,mandatory_count=121,
        feature_model_class=operator['native_model_class'],feature_attention_class=operator['native_attention_class'],
        source_code_sha256=TRAIN_HEAD_SOURCE_SHA256,shards=shards,dataset_generation_seconds=time.perf_counter()-start)
    write(manifest_path,manifest)
    return root


def main():
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['t3','linearno'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    pipeline_started=time.perf_counter()
    source_snapshot=OUT/'config/head_execution_sources'/TRAIN_HEAD_SOURCE_SHA256
    source_snapshot.mkdir(parents=True,exist_ok=True);(source_snapshot/'train_head.py').write_bytes(TRAIN_HEAD_SOURCE_BYTES)
    status_path=OUT/f'runtime/{a.architecture}_head_substages.json'
    state=dict(architecture=a.architecture,pid=os.getpid(),execution_status='running',stages={},
               actual_order=['preliminary_evaluation','feature_bank','head_training','head_acceptance'],
               execution_source_sha256=TRAIN_HEAD_SOURCE_SHA256,source_snapshot=str(source_snapshot))
    # Send the non-CA cohort first when reconstruction has already been accepted.
    gate=OUT/'config/evaluation_gate.json'
    evaluator=OUT/'code/evaluate.py'
    if gate.exists() and read(gate).get('accepted') and evaluator.exists():
        command=[sys.executable,'-u',str(evaluator),'--architecture',a.architecture,'--phase','preliminary']
        log=OUT/f'logs/{a.architecture}_preliminary.log';started=time.perf_counter()
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=CURRENT,stdout=stream,stderr=subprocess.STDOUT)
            state.update(active_stage='preliminary_evaluation');state['stages']['preliminary_evaluation']=dict(execution_status='running',command=command,pid=process.pid,log=str(log))
            write(status_path,state);process.wait()
        state['stages']['preliminary_evaluation'].update(execution_status='completed' if process.returncode==0 else 'failed',returncode=process.returncode,seconds=time.perf_counter()-started)
        write(status_path,state)
        if process.returncode:write(OUT/f'results/{a.architecture}_preliminary_attempt.json',dict(returncode=process.returncode,head_training_independent=True))
    state.update(active_stage='feature_bank');state['stages']['feature_bank']=dict(execution_status='running');write(status_path,state)
    bank_started=time.perf_counter();bank=build_bank(a.architecture);bank_seconds=time.perf_counter()-bank_started
    output=OUT/'checkpoints'/a.architecture/'head';output.mkdir(exist_ok=True)
    state['stages']['feature_bank'].update(execution_status='completed',seconds=bank_seconds,
        feature_generation_seconds=read(bank/'manifest.json')['dataset_generation_seconds'])
    state.update(active_stage='head_training');state['stages']['head_training']=dict(execution_status='running');write(status_path,state)
    head_started=time.perf_counter()
    result=train_pressure_head(bank,output,'pressure_d8',feature_dim=8,device='cuda:0',epochs=200,patience=201,batch_points=65536,seed=0)
    head_seconds=time.perf_counter()-head_started
    result.update(backbone=a.architecture,execution_status='completed',evidence_status='pending',
                  data_scope='own backbone frozen features; same900train/100validation; no test labels',
                  truth_access_scope='inference acquisition only; head training uses train pressure labels and validation selection',
                  dataset_generation_seconds=read(bank/'manifest.json')['dataset_generation_seconds'],
                  head_bank_stage_seconds=bank_seconds,head_training_stage_seconds=head_seconds,
                  train_head_code_sha256=TRAIN_HEAD_SOURCE_SHA256)
    write(output/'manifest.json',result)
    state['stages']['head_training'].update(execution_status='completed',seconds=head_seconds,training_loop_seconds=result['training_seconds'])
    state.update(active_stage='head_acceptance');state['stages']['head_acceptance']=dict(execution_status='running');write(status_path,state)
    acceptance_started=time.perf_counter()
    import validate_head
    validate_head.validate(a.architecture)
    state['stages']['head_acceptance'].update(execution_status='completed',seconds=time.perf_counter()-acceptance_started)
    state.update(execution_status='completed',active_stage=None,pipeline_including_preliminary_seconds=time.perf_counter()-pipeline_started)
    write(status_path,state)


if __name__=='__main__':main()
