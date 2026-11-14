"""Interface, source-variant, normalization and compensation checks."""
import json
import types
import numpy as np
import torch
from common import OUT,Dataset,build_model,install,seeds,write,relative_error

torch.set_num_threads(4);seeds(0)
data=Dataset();sid=data.split['training_ids'][0];pos,fx,truth=data.sample(sid)
checks=[]
for name in ['Transolver3','LinearNO']:
    model=build_model(name).eval()
    with torch.no_grad():
        for count in [7225,2709,3612,5419]:
            y=model(pos[:count][None],fx[:count][None]);assert y.shape==(1,count,1) and torch.isfinite(y).all()
            checks.append(dict(backbone=name,points=count,shape=list(y.shape),finite=True))
        sample_pos=pos[:32][None];sample_fx=fx[:32][None]
        native=model(sample_pos,sample_fx);context=install(name,model)
        context.set(torch.ones(32));unit=model(sample_pos,sample_fx)
        torch.testing.assert_close(unit,native,atol=2e-6,rtol=2e-5)
        # Exact weighted-duplication identity is applied to the first fixed-input
        # LinearNO attention layer, avoiding later-layer changed features.
        if name=='LinearNO':
            context.clear();layer=model.blocks[0].Attn
            hidden=model.blocks[0].ln_1(model.preprocess(torch.cat((sample_pos,sample_fx),dim=-1)))
            multiplicity=torch.tensor([1,2,3,2]*8,dtype=torch.float32)
            context.set(multiplicity);weighted=layer(hidden)
            context.clear();repeated=torch.repeat_interleave(hidden,multiplicity.long(),dim=1)
            duplicated=layer(repeated)
            offsets=torch.cat((torch.tensor([0]),multiplicity.long().cumsum(0)[:-1]))
            torch.testing.assert_close(weighted,duplicated[:,offsets],atol=2e-6,rtol=2e-5)
        context.clear()
    checks.append(dict(backbone=name,parameter_count=sum(p.numel() for p in model.parameters()),
                       unit_compensation_max_abs=float((unit-native).abs().max())))
conv=build_model('LinearNO',conv=True).eval()
with torch.no_grad():
    y=conv(pos[None],fx[None]);assert y.shape==(1,7225,1) and torch.isfinite(y).all()
    error=None
    try:conv(pos[:2709][None],fx[:2709][None])
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    assert error is not None,'Expected native convolution grid constraint'
checks.append(dict(backbone='LinearNO',variant='conv_temp',full_interface=True,
                   sampled_points=2709,sampled_interface_error=error,
                   parameter_count=sum(p.numel() for p in conv.parameters())))
write(OUT/'results/interface_preflight.json',dict(status='passed',checks=checks,
      selected_variable_point_class='official LinearAttentionNeuralOperator with LinearNO_temp blocks',
      additional_full_only_class='official LinearAttentionNeuralOperator with LinearNO_Conv_temp blocks',
      existing_conv_body_modified=False,normalizer_fit_ids=data.split['training_ids'],
      test_in_model_selection=False))
print(json.dumps(checks,indent=2),flush=True)
