from ByrdLab.library.RandomNumberGenerator import torch_rng
import torch
from ByrdLab import FEATURE_TYPE

class ZeroInitialize():
    def __call__(self, model, fix_init_model=False, seed=100):
        for param in model.parameters():
            param.data.zero_()
            param.data.add_(1)
    

class RandomInitialize():
    def __init__(self, scale=6):
        self.scale = scale
    def __call__(self, model, fix_init_model=False, seed=100):
        rng = torch_rng(seed=seed) if fix_init_model else None
        for param in model.parameters():
            init_param = self.scale * torch.randn(
                param.size(), dtype=FEATURE_TYPE, generator=rng
            )
            param.data.copy_(init_param)