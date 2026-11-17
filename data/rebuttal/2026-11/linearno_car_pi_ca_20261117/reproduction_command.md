# Reproduction and public verification

From the paper-repository root, validate the public numeric records and head artifact without a GPU:

```sh
python3 reproducibility/rebuttal/2026-11/linearno_car_pi_ca_20261117/verify_published.py --data-dir data/rebuttal/2026-11/linearno_car_pi_ca_20261117
```

Full execution uses the recorded workspace's E3 frozen source snapshot, official LinearNO checkpoint, E5 case split/normalizer and ShapeNet-Car data. Their paths and identities are in checkpoint_registry.csv and source_hashes.json. The official upstream is https://github.com/HiPRL/LinearNO at3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269; its full implementation is not redistributed here. Python3.10, PyTorch2.10.0+cu128 and CUDA12.8 were used; exact environment/device records are in the inference manifests.

Copy this reproduction directory to a fresh sibling of the original experiment directory, under `07_current/linearno_followup_20261114/reuse_review/<fresh_name>`. This retains the original relative workspace resolver. The frozen E3/E5 inputs and recorded external data must be present. Completed output folders are protected against overwrite.

From that fresh directory:

```sh
python3 bind_splits.py --manifest /mnt/ai-material-data/iclr2027_adatransolver/07_current/rebuttal_20261112/experiments/E3_backbones/checkpoints/seed0/shapenet_car_linearno_ddp2_e400_trainingseed0_manifest.json --output splits/seed0.json
python3 workflow.py --group head_adaptive --gpu 0
python3 workflow.py --group controls --gpu 3
python3 validate_trained_head.py --gpu 4
python3 verify_accuracy.py
python3 benchmark_inference.py --target field --gpu 4 --cpu-set 48,49,50,51
python3 benchmark_inference.py --target cd --gpu 5 --cpu-set 52,53,54,55
python3 package_results.py
```

Choose currently available GPUs/CPU cores and preserve their actual identities in the output. Head/adaptive and control workflows may run concurrently on separate GPUs. Each target's formal timing requires its exclusive GPU. The commands above preserve20 warmups and100 formal repetitions. The relevant current inference entry is `LinearNOCarRequest` in inference_request.py; older preparation helpers are retained solely to satisfy the frozen source lock and are not B/C timing evidence for this package.

The supplied `linearno_car_pressure_head_seed0.pt` and relative-path head_inference_manifest.json can be loaded with `LinearNOCA` alongside the exact backbone and normalizer in checkpoint_registry.csv. PI uses no trained head. The included backbone adapter/evaluation bridge references the external pinned native implementation.
