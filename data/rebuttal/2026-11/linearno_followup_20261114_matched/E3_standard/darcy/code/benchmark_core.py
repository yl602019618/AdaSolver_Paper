"""Dataset-independent persistence around the shared direct-wall timing helper."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
from timing_helper import measure_paired


def run_group(directory,group_id,requests,metadata,*,scope,check_output,identity_of,guard=None,extra_profiles=None):
    """All identity inspection happens outside the timed callable interval."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    destination=directory/(group_id+'.json')
    def persist(value):
        temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(destination)
    def finish_profiles(result):
        if not extra_profiles:return result
        if result.get('profile_execution_status')=='completed':return result
        if guard:guard()
        try:
            result['profiles']=extra_profiles(requests)
            if guard:guard()
            result['profile_execution_status']='completed';result.pop('profile_error',None)
        except Exception as exc:
            result['profile_execution_status']='failed';result['profile_error']=repr(exc);persist(result);raise
        persist(result);return result
    if destination.exists():
        old=json.loads(destination.read_text())
        if old.get('execution_status')!='completed':
            raise FileExistsError('incomplete group must be archived before explicit recovery')
        if old.get('metadata')!=metadata:raise RuntimeError('existing timing metadata differs from current request')
        if old.get('profile_execution_status')!='completed' and extra_profiles:
            for name,request in requests.items():check_output(name,request())
        return finish_profiles(old)
    identity_rows=[];first_identity={}
    def check(name,output):
        identity=identity_of(name,requests[name]);first_identity[name]=identity
        return check_output(name,output)
    def progress(kind,current,total):
        for name,request in requests.items():
            identity=identity_of(name,request)
            if identity!=first_identity[name]:raise RuntimeError('point identity changed within fixed-seed timing condition')
            identity_rows.append(dict(condition=name,kind=kind,repeat_id=current-1,**identity))
        (directory/'heartbeat.json').write_text(json.dumps(dict(group=group_id,kind=kind,completed=current,total=total,time=time.time()))+'\n')
    seed=int.from_bytes(hashlib.sha256(group_id.encode()).digest()[:4],'little')
    result=measure_paired(requests,scope=scope,warmups=20,repeats=100,order_seed=seed,
                          check_output=check,guard=guard,progress=progress)
    result.update(execution_status='completed',group_id=group_id,metadata=metadata,identities=identity_rows,
                  verification='all first outputs matched accepted accuracy; all timed identities remained equal')
    result['profile_execution_status']='pending' if extra_profiles else 'not_requested'
    persist(result)
    return finish_profiles(result)
