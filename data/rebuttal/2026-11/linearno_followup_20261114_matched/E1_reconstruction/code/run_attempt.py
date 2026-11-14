"""Worker attempt receipts and failure evidence; no timed-path instrumentation."""
from __future__ import annotations
import fcntl,hashlib,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
_CURRENT=None


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def preserve_existing(path):
    if _CURRENT is not None:_CURRENT.preserve(Path(path))


def track_status(path):
    if _CURRENT is not None:_CURRENT.status_path=Path(path)


def read_status(path):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {}
    except json.JSONDecodeError:
        preserve_existing(path)
        return {'execution_status':'invalid_prior_status_json','state_recovery':'reconstruct progress from raw records; original status archived'}


class Attempt:
    def __init__(self,stage,*,root=ROOT):
        self.root=Path(root);self.stage=stage;self.status_path=None;self.lock=None
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.directory=self.root/'runtime/attempts'/f'{stage}_{stamp}_{os.getpid()}'
        self.directory.mkdir(parents=True,exist_ok=False)
        self.record=dict(stage=stage,pid=os.getpid(),parent_pid=os.getppid(),started_at_utc=datetime.now(timezone.utc).isoformat(),
            started_at_unix=time.time(),execution_status='starting',command=[sys.executable,*sys.argv],
            environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
            preserved_artifacts=[],partial_evidence=[])
        self.flush()

    def flush(self):atomic_json(self.directory/'attempt.json',self.record)

    def preserve(self,path):
        if not path.is_file():return
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        ident=hashlib.sha256(str(path).encode()).hexdigest()[:12]
        dest=self.directory/'prior'/f'{ident}_{path.name}';dest.parent.mkdir(exist_ok=True)
        if dest.exists():return  # Keep the earliest prior version in this attempt.
        dest.write_bytes(data)
        self.record['preserved_artifacts'].append(dict(source=str(path),archive=str(dest.relative_to(self.directory)),sha256=digest));self.flush()

    def __enter__(self):
        global _CURRENT
        locks=self.root/'runtime/worker_locks';locks.mkdir(parents=True,exist_ok=True)
        self.lock=(locks/(self.stage+'.lock')).open('a')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.record.update(execution_status='refused_duplicate_worker',finished_at_unix=time.time());self.flush();self.lock.close()
            raise RuntimeError(f'An existing {self.stage} worker still holds its lock; do not duplicate it')
        _CURRENT=self;self.record['execution_status']='running';self.flush();return self

    def _salvage(self,tb):
        serial=0
        while tb is not None:
            frame=tb.tb_frame;name=frame.f_code.co_name;file=Path(frame.f_code.co_filename).name
            keys=(['raw','first','names','scope','repeat','kind'] if name=='measure_paired' else
                  ['checks','profiles','cache_records','key','state'] if file in ['car_timing.py','car_accuracy.py'] else
                  ['rows','prefix'] if file=='car_memory.py' else
                  ['state','errors'] if file=='gpu_monitor.py' else [])
            values={k:frame.f_locals[k] for k in keys if k in frame.f_locals}
            if values:
                try:encoded=json.dumps(values,indent=2,allow_nan=False)
                except (TypeError,ValueError):encoded=None
                if encoded is not None:
                    path=self.directory/f'partial_{serial}_{file}_{name}.json';path.write_text(encoded+'\n')
                    self.record['partial_evidence'].append(dict(file=path.name,frame=file+':'+name));serial+=1
            tb=tb.tb_next

    def __exit__(self,kind,error,tb):
        global _CURRENT
        try:
            self.record.update(finished_at_unix=time.time(),finished_at_utc=datetime.now(timezone.utc).isoformat())
            if kind is None:self.record['execution_status']='completed'
            else:
                self.record.update(execution_status='failed',exception=kind.__name__,message=str(error))
                (self.directory/'traceback.txt').write_text(''.join(traceback.format_exception(kind,error,tb)))
                self._salvage(tb)
                if self.status_path is not None:
                    self.preserve(self.status_path);status=read_status(self.status_path)
                    status.update(execution_status='failed',evidence_status='pending',attempt_receipt=str(self.directory/'attempt.json'),
                                  failure_type=kind.__name__,failure_message=str(error))
                    atomic_json(self.status_path,status)
            self.flush()
        finally:
            _CURRENT=None
            if self.lock is not None:fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
        return False
