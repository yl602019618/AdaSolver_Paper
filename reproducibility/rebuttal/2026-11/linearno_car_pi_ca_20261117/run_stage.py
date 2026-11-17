"""Run one scheduler-admitted Car LinearNO head or evaluation stage."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, E3, sha256, verify_lock, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--stage", choices=("head_dataset", "head", "full", "adaptive"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    verify_lock()
    task_lock = open(ROOT / f".seed{args.seed}_{args.stage}.lock", "a")
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not json.loads((ROOT / "interface_validation.json").read_text())["passed"]:
        raise ValueError("CPU interface validation has not passed")
    name = f"shapenet_car_linearno_ddp2_e400_trainingseed{args.seed}"
    operator_manifest = (E3 if args.seed == 0 else ROOT) / "checkpoints" / f"seed{args.seed}" / f"{name}_manifest.json"
    operator = json.loads(operator_manifest.read_text())
    if operator["seed"] != args.seed or operator["epochs"] != 400 or operator["total_optimization_steps"] != 142000:
        raise ValueError("The requested matched backbone is incomplete")
    if sha256(operator["checkpoint"]) != operator["checkpoint_sha256"]:
        raise ValueError("Backbone checkpoint hash differs")
    lock = open(f"/tmp/adasolver_profile_gpu_{args.gpu}.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupancy = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if occupancy.strip():
        raise RuntimeError("RUNNING_IN_OTHER_SESSION: requested GPU is occupied")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"}
    splits = ROOT / "splits" / f"seed{args.seed}.json"
    subprocess.run([sys.executable, str(ROOT / "bind_splits.py"), "--manifest", str(operator_manifest),
                    "--output", str(splits)], env=env, check=True)
    bank = ROOT / "records" / f"head_dataset_seed{args.seed}"
    head = ROOT / "checkpoints" / f"seed{args.seed}" / "head"
    head_name = f"ca_pressure_linearno_trainingseed{args.seed}_headseed{args.seed}"
    head_manifest = head / f"{head_name}_manifest.json"
    if args.stage == "head_dataset":
        command = [sys.executable, str(ROOT / "prepare_head_dataset.py"), "--manifest", str(operator_manifest),
                   "--splits-path", str(splits), "--output", str(bank), "--device", "cuda:0"]
    elif args.stage == "head":
        command = [sys.executable, str(ROOT / "train_head.py"), "--dataset", str(bank),
                   "--output-dir", str(head), "--name", head_name, "--seed", str(args.seed), "--device", "cuda:0"]
    else:
        output = ROOT / "records" / f"seed{args.seed}" / args.stage
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Preserve existing evaluation output: " + str(output))
        command = [sys.executable, str(ROOT / "evaluate.py"), "--manifest", str(operator_manifest),
                   "--splits", str(splits), "--split", "test", "--device", "cuda:0", "--output", str(output)]
        if args.stage == "full":
            if args.seed == 0:
                raise ValueError("Reuse the verified E3 seed0 Full reference; do not duplicate it")
            command.extend(["--method", "fullmesh", "--seeds", "0"])
        else:
            config = ROOT / ("seed0_missing_methods.json" if args.seed == 0 else "evaluation_methods.json")
            command.extend(["--head-manifest", str(head_manifest), "--configs", str(config), "--seeds", "0", "1", "2"])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime = ROOT / "runs" / f"seed{args.seed}"
    runtime.mkdir(parents=True, exist_ok=True)
    log = runtime / f"{args.stage}_{stamp}.log"
    state_path = runtime / f"{args.stage}_launch.json"
    state = {"status": "RUNNING", "training_seed": args.seed, "head_seed": args.seed,
             "stage": args.stage, "physical_gpu": args.gpu, "command": command,
             "log": str(log), "launcher_pid": os.getpid(), "started_at_utc": stamp,
             "operator_sha256": operator["checkpoint_sha256"]}
    start = time.monotonic()
    with log.open("x") as stream:
        child = subprocess.Popen(command, cwd=runtime, env=env, stdout=stream, stderr=subprocess.STDOUT)
        state["pid"] = child.pid
        write_json(state_path, state)
        code = child.wait()
    state.update(status="COMPLETED" if code == 0 else "FAILED", returncode=code,
                 process_wall_seconds=time.monotonic()-start,
                 reserved_gpu_seconds=time.monotonic()-start,
                 finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(state_path, state)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
