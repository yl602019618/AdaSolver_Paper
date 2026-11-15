"""Schedule complete backbone, head and evaluation jobs on currently idle GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, write


def complete(benchmark, seed, stage):
    base = ROOT / "checkpoints" / benchmark / f"seed{seed}"
    path = (base / stage / "manifest.json" if stage != "evaluate" else
            ROOT / "records" / benchmark / f"seed{seed}" / "completion.json")
    return path.exists() and json.loads(path.read_text()).get("status") == "completed"


def idle_gpus():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    return [int(row[0]) for line in result.splitlines()
            if (row := [x.strip() for x in line.split(",")]) and int(row[1]) < 64 and int(row[2]) <= 1]


def main():
    benchmarks = ("pipe", "darcy", "elasticity")
    stages = ("backbone", "head", "evaluate")
    active, failed = {}, []
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    while True:
        for key, entry in list(active.items()):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["stream"].close()
            del active[key]
            if code != 0 or not complete(*key):
                failed.append({"benchmark": key[0], "seed": key[1], "stage": key[2],
                               "exit_code": code, "log": entry["log"], "gpu": entry["gpu"]})
            print("FINISHED", key, "exit", code, flush=True)
        finished = [(b, seed, stage) for b in benchmarks for seed in (0, 1, 2)
                    for stage in stages if complete(b, seed, stage)]
        failed_keys = {(x["benchmark"], x["seed"], x["stage"]) for x in failed}
        available = [gpu for gpu in idle_gpus() if gpu not in {x["gpu"] for x in active.values()}]
        candidates = []
        for stage in stages:
            script = ROOT / "code" / ("train_" + stage + ".py" if stage != "evaluate" else "evaluate.py")
            if not script.exists():
                continue
            for benchmark in benchmarks:
                if not (ROOT / "data" / benchmark / "manifest.json").exists():
                    continue
                for seed in (0, 1, 2):
                    key = benchmark, seed, stage
                    prerequisites = stages[:stages.index(stage)]
                    if key not in active and key not in failed_keys and not complete(*key) and all(
                            complete(benchmark, seed, previous) for previous in prerequisites):
                        candidates.append((key, script))
        for gpu, (key, script) in zip(available, candidates):
            benchmark, seed, stage = key
            log = ROOT / "logs" / f"{benchmark}_seed{seed}_{stage}.log"
            stream = log.open("a")
            command = [sys.executable, str(script), "--benchmark", benchmark,
                       "--seed", str(seed), "--gpu", str(gpu)]
            process = subprocess.Popen(command, cwd=ROOT.parent, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = {"process": process, "gpu": gpu, "log": str(log), "stream": stream,
                           "started_at": time.time(), "command": command}
            print("STARTED", key, "gpu", gpu, "pid", process.pid, flush=True)
        status = {
            "status": "completed" if len(finished) == 27 else "running",
            "finished_stages": [list(x) for x in finished], "total_stages": 27,
            "active": [{"benchmark": k[0], "seed": k[1], "stage": k[2], "gpu": v["gpu"],
                        "pid": v["process"].pid, "log": v["log"], "started_at": v["started_at"]}
                       for k, v in active.items()], "failed": failed, "updated_at": time.time(),
        }
        write(ROOT / "runtime/supervisor.json", status)
        if len(finished) == 27:
            return
        if failed and not active and not candidates:
            status["status"] = "failed"
            write(ROOT / "runtime/supervisor.json", status)
            raise SystemExit(1)
        time.sleep(15)


if __name__ == "__main__":
    main()
