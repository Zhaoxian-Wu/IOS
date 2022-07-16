import random
import torch

def random_rng(seed=10):
    return random.Random(seed)

def torch_rng(seed=10):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
