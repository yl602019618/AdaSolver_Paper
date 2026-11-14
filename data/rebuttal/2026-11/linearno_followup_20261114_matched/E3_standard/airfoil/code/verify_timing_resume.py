"""CPU failure/recovery verification of record durability; no scientific latency claim."""
from common import *
from evaluate import attach_reconstruction
api,gate=attach_reconstruction()
from benchmark_core import run_group
import tempfile

class RequestFixture:
    def __init__(self):self.calls=0
    def __call__(self):self.calls+=1;return np.array([1.],dtype=np.float64)


def main():
    original=torch.cuda.synchronize;torch.cuda.synchronize=lambda:None
    metadata={name:{field:'development_only' for field in ['checkpoint_id','normalizer_id','reconstruction_id','request_code_sha256','runtime_code_sha256','reconstruction_sha256']} for name in ['old','new']}
    path=OUT/'results/benchmark_development';path.mkdir(exist_ok=True)
    def check(name,output):assert output[0]==1.;return {'passed':True}
    def fail(requests):raise ValueError('intentional separate-profile failure')
    try:
        with tempfile.TemporaryDirectory(dir=path,prefix='core_resume_') as folder:
            group_dir=Path(folder)/'groups'
            initial={name:RequestFixture() for name in metadata}
            try:run_group(group_dir,'development_only',initial,metadata,scope='CPU development fixture',check_output=check,identity_of=lambda n,r:{'fixed':True},extra_profiles=fail)
            except ValueError as e:assert str(e)=='intentional separate-profile failure'
            saved=read(group_dir/'development_only.json');assert saved['execution_status']=='completed' and saved['profiles_status']=='failed'
            assert len(saved['raw'])==240 and all(r.calls==121 for r in initial.values())
            recovered={name:RequestFixture() for name in metadata}
            after=run_group(group_dir,'development_only',recovered,metadata,scope='CPU development fixture',check_output=check,identity_of=lambda n,r:{'fixed':True},extra_profiles=lambda r:{n:[] for n in r})
            assert after['raw']==saved['raw'] and all(r.calls==1 for r in recovered.values()) and after['profiles_status']=='completed'
        write(OUT/'results/timing_resume_development_validation.json',dict(passed=True,direct_observations_preserved=240,
            repeated_direct_measurements_on_recovery=0,cache_rehydration_requests_per_condition=1,
            scope='CPU fixture with synchronization stub, intentional profile failure and exact raw-record preservation; not model timing',
            timing_core_source_sha256=sha(OUT/'code/benchmark_core.py')))
    finally:torch.cuda.synchronize=original


if __name__=='__main__':main()
