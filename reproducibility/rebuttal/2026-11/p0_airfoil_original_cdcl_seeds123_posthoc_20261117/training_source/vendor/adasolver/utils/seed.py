from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    return torch.Generator().manual_seed(seed)


def make_numpy_rng(seed: int, sample_id: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, sample_id]))
