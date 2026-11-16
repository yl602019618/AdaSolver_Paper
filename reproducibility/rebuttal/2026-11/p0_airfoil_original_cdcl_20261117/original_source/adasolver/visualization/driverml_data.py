"""Small, deterministic surface snapshots from the actual evaluation tensors."""
from pathlib import Path
import json
import numpy as np
from adasolver.inference.driverml_contract import rng_for
from adasolver.inference.driverml_sampling import normalize_score


def export_snapshot(sample,selected,surface,pressure_full,score,record,directory,display_count=50000):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    n=sample.point_count
    display=rng_for(0,sample.run_id,80).choice(n,min(display_count,n),replace=False)
    sampled=rng_for(record['seed'],sample.run_id,81,int(record['budget_fraction']*1000)).choice(len(selected.indices),min(8000,len(selected.indices)),replace=False)
    source=surface.reference_to_selected[display]
    label=f"{record['metric_target']}_{record['method']}_run{sample.run_id}_s{record['seed']}_b{record['budget_fraction']:.3f}"
    np.savez_compressed(directory/f'{label}.npz',xyz=sample.pos.numpy()[display],
         pressure_gt=sample.pressure.numpy()[display],pressure_pred=pressure_full.numpy()[display],
         relative_density=(selected.sampling_density[source]*n).astype(np.float32),
         acquisition_score=np.zeros(len(display),dtype=np.float32) if score is None else normalize_score(score)[display].astype(np.float32),
         selected_xyz=sample.pos.numpy()[selected.indices[sampled]],
         is_coarse=sampled<record['coarse_count'],display_indices=display)
    metadata=dict(record,display_count=len(display),displayed_selected_count=len(sampled),display_seed_rule='SeedSequence([0,run_id,80,0])')
    (directory/f'{label}.json').write_text(json.dumps(metadata,indent=2)+'\n')


def export_fullmesh(sample,pressure,record,directory,display_count=50000):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    display=rng_for(0,sample.run_id,80).choice(sample.point_count,min(display_count,sample.point_count),replace=False)
    np.savez_compressed(directory/f'fullmesh_run{sample.run_id}.npz',xyz=sample.pos.numpy()[display],
                        normal_x=sample.normal.numpy()[display,0],pressure_gt=sample.pressure.numpy()[display],
                        pressure_pred=pressure.numpy()[display])
    (directory/f'fullmesh_run{sample.run_id}.json').write_text(json.dumps(record,indent=2)+'\n')
