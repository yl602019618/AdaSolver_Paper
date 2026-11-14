from pathlib import Path
import json,hashlib,datetime
from car_request import configuration_bank,E3,E5,OUT

bank=configuration_bank();base=json.loads((E3/'config/latency_protocol.json').read_text())
split=json.loads((E3/'config/splits_seed0.json').read_text());v=split['operator_validation']
protocol=dict(protocol_id='car_reconstruction_followup_v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    dataset='shapenet_car',test_ids=split['test'],validation_ids=[v[0],v[len(v)//2],v[-1]],
    timing_cases=base['case_ids'],sampling_seeds=[0,1,2],budgets=[.375,.5,.75],
    configurations=bank,
    accuracy_variants={'Transolver3':['full','uniform','pi','cpg_beta0','cpg_beta1','ca'],'LinearNO':['full','uniform','cpg_beta0','cpg_beta1']},
    timing_variants={'Transolver3':['full','uniform','pi','cpg_beta1','ca'],'LinearNO':['full','uniform','cpg_beta1']},
    expected_accuracy_requests=7400,expected_old_new_metric_rows=14800,
    scope_A='CPU model input and CPU complete float64 field; dynamic map built each request',
    scope_B='resident GPU model inputs and GPU complete float64 field; CPU geometry metadata/selection and transfers still included',
    scope_C='A boundary; only identical geometry/point-set map is reused; scores and model predictions recomputed',
    warmups=20,repeats=100,stage_repeats=5,physical_gpu=6,cpu_affinity=list(range(128,136)),threads=4,
    backend='native E3 T3 matrix / official LinearNO; eager FP32; no key-softmax-layout optimization',
    ca='T3 seed0 matching200-epoch head; eager head+within-request first-block reuse. LinearNO Car CA unavailable, not borrowed.',
    full_alias='one real series per backbone/case/scope; old/new share this series; Full has no sampling seed',
    cpg_beta_pair='same coarse values and selected hashes required; beta0/1 both evaluated, primary timing fixedbeta1',
    reuse_counts=[1,3,10,100],
    t3_manifest=base['t3_manifest'],linearno_manifest=base['linearno_manifest'],
    head_manifest=str(E5/'checkpoints/seed0/head/ca_pressure_trainingseed0_headseed0_manifest.json'),
    old_reconstruction='car_scipy_idw4_native_v1',new_reconstruction='car_ckdtree4_sparse_apply_v1p1',
    statistical_scope='All100 historical test cases; sampling SD across3 seeds; no new training seed claim',
    main_metric='velocity relative L2 +0.5*surface-pressure relative L2',
    supplemental_metric='Cd relative absolute error from the same Field-target strategy, force integration outside timed field request')
p=OUT/'config/car_protocol.json'
if p.exists():
    previous=json.loads(p.read_text());protocol['created_at_utc']=previous['created_at_utc'];assert previous==protocol
else:p.write_text(json.dumps(protocol,indent=2)+'\n')
print('Prepared Car protocol:7400 predictions, complete old/new reconstruction comparisons; timing A/B/C follows model readiness and GPU ownership.')
