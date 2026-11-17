"""Car CA using the frozen official LinearNO's own first-block features.

The graph input, placeholder, all operator blocks and checkpoint keys remain
unchanged. Final inference recomputes the official operator with its final
point weights; it does not use the Transolver slice cache.
"""
from common import activate_sources
activate_sources()

import torch
from linearno_adapter import LinearNOAdapter
from adasolver.inference.ca.shapenet_car import ShapeNetCarCA
from adasolver.inference.shapenet_car_contract import coarse_count


@torch.no_grad()
def coarse_features(operator, pos, fx, *, capture=False, nfull):
    if not isinstance(operator, LinearNOAdapter):
        raise TypeError("This feature path requires the official E3 LinearNO adapter")
    if pos.ndim != 3 or pos.shape[-1] != 3 or fx.shape != (*pos.shape[:2], 7):
        raise ValueError("Expected pos[B,N,3] and normalized native fx[B,N,7]")
    if pos.shape[1] != coarse_count(nfull):
        raise ValueError("Car CA requires the actual 10% selected set")
    net = operator.model
    if net.training or any(p.requires_grad for p in net.parameters()):
        raise ValueError("CA features must come from a frozen evaluation backbone")
    context = getattr(net.blocks[0].Attn, "_linear_compensation_context", None)
    if context is not None and context.weights is not None:
        raise ValueError("The coarse CA pass uses uniform measure")
    # Match the native graph forward's per-example execution and placeholder.
    features = torch.stack([net.blocks[0](net.preprocess(x[None]) + net.placeholder[None, None])[0]
                            for x in fx])
    return features, None


class LinearNOCA(ShapeNetCarCA):
    def __init__(self, operator, head_manifest, binding, device="cpu", **kwargs):
        if not isinstance(operator, LinearNOAdapter):
            raise TypeError("LinearNO CA cannot load a different backbone")
        super().__init__(operator, head_manifest, binding, device, use_cuda_graph=False)
        if self.metadata.get("backbone") != "LinearNO" or self.metadata.get("feature_backend") != "official_linearno_native_block0":
            raise ValueError("A head trained on this LinearNO feature path is required")

    @torch.no_grad()
    def feature_pass(self, pos, fx, nfull, *, capture=True):
        return coarse_features(self.operator, pos, fx, capture=capture, nfull=nfull)

    @torch.no_grad()
    def prepare_acquisition(self, pos, fx, nfull):
        if pos.shape[1] != coarse_count(nfull):
            raise ValueError("Car CA requires the actual 10% selected set")

    @torch.no_grad()
    def acquire(self, pos, fx, nfull, *, capture=True):
        return self._acquire_eager(pos, fx, nfull, capture=capture)

    @torch.no_grad()
    def final(self, pos, fx, coarse, final, cache, compensation):
        # Recompute global key/value statistics on the final set. No coarse
        # global feature or prediction is reused as a final prediction.
        context = getattr(self.operator.model.blocks[0].Attn, "_linear_compensation_context", None)
        if context is None:
            if compensation is not None:
                raise ValueError("Install the official LinearNO measure hook before final inference")
            return self.operator(pos, fx)
        previous = context.weights
        context.set(compensation)
        try:
            return self.operator(pos, fx)
        finally:
            context.set(previous)
