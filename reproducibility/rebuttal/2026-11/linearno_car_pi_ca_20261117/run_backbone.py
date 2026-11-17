"""Print or execute an isolated, matched DDP2 Car LinearNO training command.

No GPU process starts unless --execute is supplied. The parent task schedules
admission after checking other sessions. This runner rejects occupied GPUs.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, E3, PAPER_COMMIT, activate_sources, sha256, verify_lock, write_json


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--gpu-pair", required=True)
    parser.add_argument("--cpu-set", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    protocol = verify_lock()
    gpus = [int(x) for x in args.gpu_pair.split(",")]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("Exactly two distinct physical GPUs are required")
    task = next(x for x in protocol["training_tasks"] if x["training_seed"] == args.seed)
    name = task["name"]
    output = ROOT / "checkpoints" / f"seed{args.seed}"
    cwd = ROOT / "runs" / f"seed{args.seed}"
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
               "--nproc_per_node=2", "--master_addr=127.0.0.1", f"--master_port={args.port}",
               str(E3 / "code/train_linearno.py"), "--data-path", protocol["data_path"],
               "--normalizer", protocol["normalizer"], "--output-dir", str(output),
               "--name", name, "--epochs", "400", "--batch-size", "2",
               "--learning-rate", ".001", "--weight-decay", ".00001",
               "--max-grad-norm", ".1", "--nmin-fraction", ".4",
               "--pressure-weight", ".5", "--validation-fraction", ".1",
               "--split-seed", "20260905", "--seed", str(args.seed)]
    resume = output / f"{name}_resume.pt"
    if args.resume:
        if not resume.is_file():
            raise FileNotFoundError(resume)
        command.extend(["--resume", str(resume)])
    environment = {"CUDA_VISIBLE_DEVICES": args.gpu_pair,
                   "PYTHONPATH": os.pathsep.join([str(E3 / "code"), str(E3 / "code/vendor")]),
                   "PYTHONUNBUFFERED": "1", **{k: str(args.threads) for k in (
                       "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    prepared = {"training_seed": args.seed, "command": command, "cwd": str(cwd),
                "environment": environment, "cpu_set": args.cpu_set,
                "output_manifest": str(output / f"{name}_manifest.json"),
                "paper_commit": PAPER_COMMIT, "head_status": "not_started"}
    print(json.dumps(prepared, indent=2), flush=True)
    if not args.execute:
        return
    task_lock = open(ROOT / f".seed{args.seed}_backbone.lock", "a")
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    activate_sources()
    import torch
    if str(torch.__version__) != "2.10.0+cu128" or torch.version.cuda != "12.8":
        raise ValueError("The matched cohort uses torch 2.10.0+cu128 / CUDA 12.8")
    locks = []
    for gpu in sorted(gpus):
        lock = open(f"/tmp/adasolver_profile_gpu_{gpu}.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    devices = {int(row[0]): {"uuid": row[1].strip(), "name": row[2].strip()}
               for row in (line.split(",", 2) for line in info.splitlines())}
    selected = {gpu: devices[gpu] for gpu in gpus}
    if any(x["name"] != protocol["gpu_model"] for x in selected.values()):
        raise ValueError("GPU model differs from the locked H20 cohort")
    occupancy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    if any(x["uuid"] in occupancy for x in selected.values()):
        raise RuntimeError("RUNNING_IN_OTHER_SESSION: requested GPUs already have compute processes")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Preserve existing output {output}; use --resume for its own interrupted run")
    cpu_ids = []
    for span in args.cpu_set.split(","):
        ends = list(map(int, span.split("-")))
        cpu_ids.extend(range(ends[0], ends[-1] + 1))
    os.sched_setaffinity(0, cpu_ids)
    output.mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    state_path = cwd / "launch.json"
    state = {**prepared, "started_at_utc": now(), "execution_status": "running",
             "gpu_devices": selected, "launcher_pid": os.getpid(),
             "protocol_sha256": sha256(ROOT / "training_tasks.json"), "attempts": []}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        state["attempts"] = previous.get("attempts", [])
    attempt = {"started_at_utc": now(), "command": command, "resume": args.resume}
    state["attempts"].append(attempt)
    start = time.monotonic()
    log_path = cwd / ("resume_console.log" if args.resume else "console.log")
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(command, cwd=cwd, env={**os.environ, **environment},
                                   stdout=log, stderr=subprocess.STDOUT)
        attempt.update(pid=process.pid, log=str(log_path))
        write_json(state_path, state)
        code = process.wait()
    attempt.update(returncode=code, finished_at_utc=now(), process_wall_seconds=time.monotonic()-start)
    state.update(execution_status="completed" if code == 0 else "failed",
                 reserved_gpu_seconds=2 * sum(x.get("process_wall_seconds", 0.) for x in state["attempts"]),
                 timing_scope="two GPUs reserved; process wall includes data setup, training, validation and checkpoint I/O")
    if code == 0:
        manifest_path = Path(prepared["output_manifest"])
        manifest = json.loads(manifest_path.read_text())
        expected = {"seed": args.seed, "epochs": 400, "total_optimization_steps": 142000,
                    "world_size": 2, "global_batch_size": 2, "train_samples": 710,
                    "validation_samples": 79, "sealed_test_samples": 100,
                    "normalizer_sha256": protocol["normalizer_sha256"]}
        for key, value in expected.items():
            if manifest.get(key) != value:
                state.update(execution_status="failed_protocol", failure=key)
                write_json(state_path, state)
                raise ValueError("Completed backbone protocol mismatch: " + key)
        from linearno_adapter import LinearNOSpec
        state.update(parameter_count=sum(p.numel() for p in LinearNOSpec().build().parameters()),
                     checkpoint_sha256=sha256(manifest["checkpoint"]),
                     best_validation_field_error=manifest["best_validation_field_error"])
    write_json(state_path, state)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
