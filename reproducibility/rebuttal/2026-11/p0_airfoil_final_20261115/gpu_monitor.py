"""Admit an unused physical GPU and record utilization, clocks and ownership."""
from contextlib import AbstractContextManager
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time


class ExclusiveGPU(AbstractContextManager):
    def __init__(self,physical_index,output,interval=.1,shutdown_on_exit=True):
        self.index=int(physical_index);self.output=Path(output);self.interval=interval
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.pid=os.getpid();self.samples=[];self.stop=threading.Event();self.errors=[]
        self.shutdown_on_exit=shutdown_on_exit
        from inference_speed.nvml import NVML
        self.nvml=NVML(self.index)

    def snapshot(self):
        data,jobs=self.nvml.snapshot()
        return {"time":time.time(),"physical_gpu":self.index,"gpu":data,"processes":jobs}

    def cli_snapshot(self):
        fields="uuid,name,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu"
        result=subprocess.check_output(["nvidia-smi","-i",str(self.index),f"--query-gpu={fields}","--format=csv,noheader,nounits"],text=True)
        values=next(csv.reader(io.StringIO(result),skipinitialspace=True))
        data=dict(zip(fields.split(","),[v.strip() for v in values]))
        processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"],text=True)
        jobs=[]
        for row in csv.reader(io.StringIO(processes),skipinitialspace=True):
            if row and row[0].strip()==data["uuid"]:
                jobs.append({"pid":int(row[1]),"name":row[2].strip(),"memory_mib":row[3].strip()})
        return {"time":time.time(),"physical_gpu":self.index,"gpu":data,"processes":jobs}

    def __enter__(self):
        fd=int(os.environ["ADASOLVER_PROFILE_LOCK_FD"])
        expected=os.stat(f"/tmp/adasolver_profile_gpu_{self.index}.lock")
        actual=os.fstat(fd)
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError("Inherited GPU lock inode mismatch")
        self.lock=os.fdopen(os.dup(fd),"w")
        for _ in range(3):
            state=self.snapshot();self.samples.append(state)
            if state["processes"] or int(state["gpu"]["memory.used"])>32 or int(state["gpu"]["utilization.gpu"])>1:
                self.lock.close()
                raise RuntimeError(f"GPU {self.index} is not idle: {state}")
            time.sleep(.35)
        self.initial=self.samples[-1]
        self.thread=threading.Thread(target=self._poll,daemon=True);self.thread.start()
        return self

    def _poll(self):
        while not self.stop.wait(self.interval):
            try:
                state=self.snapshot();self.samples.append(state)
                other=[p for p in state["processes"] if p["pid"]!=self.pid]
                if other:self.errors.append({"foreign_processes":other,"time":state["time"]})
            except Exception as exc:self.errors.append({"monitor_error":str(exc),"time":time.time()})

    def assert_exclusive(self):
        state=self.snapshot();self.samples.append(state)
        other=[p for p in state["processes"] if p["pid"]!=self.pid]
        if other:self.errors.append({"foreign_processes":other,"time":state["time"]})
        if self.errors:raise RuntimeError(f"GPU occupancy monitoring invalidated this run: {self.errors}")

    def __exit__(self,*exc):
        self.stop.set();self.thread.join(timeout=10)
        try:self.assert_exclusive()
        finally:
            result={"pid":self.pid,"physical_gpu":self.index,"initial_idle_checks":3,
                    "exclusive":not bool(self.errors),"errors":self.errors,"samples":self.samples,"monitor_backend":"read-only NVML, no subprocess polling"}
            self.output.write_text(json.dumps(result,indent=2)+"\n")
            self.lock.close()  # parent retains ownership until child exits
            if self.shutdown_on_exit:self.nvml.close()
