"""Durable direct timing; separate profiles cannot discard completed repeats."""
from __future__ import annotations
import hashlib,json,os,time,traceback
from pathlib import Path
from timing_helper import measure_paired


def save(path,record):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)


def run_group(directory,group_id,requests,metadata,*,scope,check_output,identity_of,guard=None,extra_profiles=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);destination=directory/(group_id+'.json')
    result=json.loads(destination.read_text()) if destination.exists() else None
    resumed=result is not None
    if result and result.get('execution_status')!='completed':raise FileExistsError('incomplete direct timing must be explicitly recovered')
    if result:
        for name,meta in metadata.items():
            for field in ['checkpoint_id','normalizer_id','reconstruction_id','request_code_sha256','runtime_code_sha256','reconstruction_sha256']:
                if result['metadata'][name][field]!=meta[field]:raise RuntimeError(f'resumed timing identity changed: {name} {field}')
    if result and result.get('profiles_status')=='completed':return result
    first_identity={};identity_rows=[]
    if result is None:
        def check(name,output):
            first_identity[name]=identity_of(name,requests[name])
            return check_output(name,output)
        def progress(kind,current,total):
            for name,request in requests.items():
                identity=identity_of(name,request)
                if identity!=first_identity[name]:raise RuntimeError('point/weight identity changed within fixed-seed condition')
                identity_rows.append(dict(condition=name,kind=kind,repeat_id=current-1,**identity))
            save(directory/'heartbeat.json',dict(group=group_id,kind=kind,completed=current,total=total,time=time.time()))
        seed=int.from_bytes(hashlib.sha256(group_id.encode()).digest()[:4],'little')
        started_at=time.time()
        try:
            result=measure_paired(requests,scope=scope,warmups=20,repeats=100,order_seed=seed,
                check_output=check,guard=guard,progress=progress)
        except Exception as exc:
            failures=directory.parent/'failures';failures.mkdir(exist_ok=True)
            save(failures/f'{group_id}.{time.time_ns()}.json',dict(execution_status='failed',group_id=group_id,
                metadata=metadata,error=str(exc),traceback=traceback.format_exc(),identities_observed=identity_rows,
                started_at_epoch=started_at,failed_at_epoch=time.time(),scope='failed group is excluded from accepted timing'))
            raise
        result.update(execution_status='completed',group_id=group_id,metadata=metadata,identities=identity_rows,
            verification='first complete field and every timed selected/coarse index and weight identity match accepted accuracy',profiles_status='pending',
            measurement_started_at_epoch=started_at,measurement_finished_at_epoch=time.time())
        save(destination,result)
    if extra_profiles:
        try:
            if guard:guard()
            if resumed:
                for name,request in requests.items():check_output(name,request())
            result['profiles']=extra_profiles(requests);result['profiles_status']='completed'
            result['profile_metadata']=metadata
            if guard:guard()
        except Exception as exc:
            result['profiles_status']='failed';result['profile_error']=dict(error=str(exc),traceback=traceback.format_exc())
            failures=directory.parent/'failures';failures.mkdir(exist_ok=True)
            save(failures/f'{group_id}.profile.{time.time_ns()}.json',dict(group_id=group_id,stage='profiles',**result['profile_error']))
            save(destination,result);raise
    else:result['profiles_status']='not_requested'
    save(destination,result);return result
