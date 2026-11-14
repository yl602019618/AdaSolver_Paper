"""Native sampling policies bound to newly trained Airfoil backbones and heads."""
from common import *
import copy
from adasolver.evaluation.airfoil2d import AirfoilEvaluator
from adasolver.data.airfoil2d_runtime import AirfoilRuntimeData
from adasolver.inference.airfoil2d_config import resolve_config
from adasolver.inference.ca.airfoil2d import Airfoil2dPressureHead
from adasolver.utils.interpolation import index_grid


class OwnPressureHead(Airfoil2dPressureHead):
    @torch.no_grad()
    def features(self,pos,fx=None):return feature0(self.operator,pos)

    def forward_with_cache(self,pos,fx=None,*,capture=False):
        if capture:raise ValueError('The frozen Field policy has operator_stage1_cache=False')
        return self.head(self.features(pos,fx)),None


def configurations(architecture,include_ca=True):
    path=OUT/'config/frozen_field_policies.json'
    bank=read(path)
    result=[]
    for name in ['uniform','pi','cpg_beta0','cpg','ca']:
        if name=='ca' and not include_ca:continue
        cfg=copy.deepcopy(bank[name])
        if name=='ca':
            cfg['head_checkpoint']=str(OUT/f'checkpoints/{architecture}/head/pressure_d8_best.pt')
            cfg['head_manifest']=str(OUT/f'checkpoints/{architecture}/head/manifest.json')
        result.append(resolve_config(cfg['method'],cfg))
    return result


class Runtime(AirfoilEvaluator):
    def __init__(self,architecture,device='cuda:0',*,model_override=None):
        self.architecture=architecture;self.device=device
        if model_override is None:self.model,self.metadata=load_model(architecture,device)
        else:self.model,self.metadata=model_override
        self.model.requires_grad_(False).eval()
        self.context,self.patched=(LinearContext(),0) if architecture=='linearno_conv' else install_compensation(self.model)
        self.manifest=str(OUT/f'checkpoints/{architecture}/manifest.json')
        self.checkpoint=self.metadata['checkpoint'];self.operator_hash=self.metadata['checkpoint_sha256']
        self.data=AirfoilRuntimeData(DATA,self.manifest)
        self.reference=index_grid((221,51))
        self.heads={};self.coarse_cache={};self.patch_cache={};self.feature_cache={};self.last_visualization=None

    def head(self,config):
        checkpoint,manifest=config['head_checkpoint'],config['head_manifest'];key=(checkpoint,manifest)
        if key not in self.heads:
            m=read(manifest)
            if m['operator_sha256']!=self.operator_hash or m['normalizer_sha256']!=self.data.normalizer_sha256:
                raise ValueError('Head must be trained on this backbone and normalizer')
            if sha(checkpoint)!=m['head_sha256']:raise ValueError('head identity changed')
            accepted=read(OUT/f'results/{self.architecture}_head_acceptance.json')
            if not accepted['passed'] or accepted['head_sha256']!=m['head_sha256']:
                raise ValueError('own-backbone head acceptance is required')
            h=OwnPressureHead(self.model,float(self.data.mean),float(self.data.std),feature_dim=8).to(self.device).eval()
            h.head.load_state_dict(torch.load(checkpoint,map_location=self.device,weights_only=True))
            self.heads[key]=(h,m['head_sha256'],m)
        return self.heads[key]

    def clear_dynamic(self):
        self.context.clear();self.coarse_cache.clear();self.feature_cache.clear();self.patch_cache.clear()

    @torch.no_grad()
    def predict_subset(self,sid,seed,budget,config):
        sampling,coarse,cache=self.sampling(sid,seed,budget,config)
        coords=self.data.coords(sid);indices=sampling.indices
        pos=torch.as_tensor(coords[indices],device=self.device)[None]
        weights=None if config['method']=='uniform' else torch.as_tensor(sampling.compensation,dtype=torch.float32,device=self.device)
        self.context.set(weights)
        try:prediction=self.data.decode(self.model(pos,pos))[0]
        finally:self.context.clear()
        metadata=dict(sampling.metadata)
        if coarse is not None and config['method'].startswith('cpg'):
            cache_key=(sid,seed,len(coarse),None,config['operator_stage1_cache'])
            values=self.coarse_cache[cache_key][0]
            metadata['coarse_prediction_hash']=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        return prediction,sampling,coarse,metadata

    @torch.no_grad()
    def predict_full(self,sid):
        pos=torch.as_tensor(self.data.coords(sid),device=self.device)[None];self.context.clear()
        return self.data.decode(self.model(pos,pos))[0]
