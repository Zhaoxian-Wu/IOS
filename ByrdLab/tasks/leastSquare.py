from ByrdLab.library.dataset import Dataset
from ByrdLab import FEATURE_TYPE, VALUE_TYPE
from ByrdLab.library.RandomNumberGenerator import torch_rng
from functools import partial

import torch

from ByrdLab.tasks import Task
from ByrdLab.library.initialize import RandomInitialize

class LeastSqaure_LinearModel(torch.nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=feature_dimension,
                                      out_features=1, bias=True)
    def forward(self, features):
        features = features.view(features.size(0), -1)
        return self.linear(features)

def train_full_batch_generator(dist_dataset, node, rng):
    while True:
        yield dist_dataset[node][:]
        
def val_full_batch_generator(dataset):
    yield dataset[:]
        
def least_square_loss(predictions, targets):
    return ((predictions-targets)**2).mean()

class LeastSquareToySet(Dataset):
    def __init__(self, set_size, dimension, noise=0.1, w_star=None,
                 fix_seed=False, seed=20):
        generator = torch_rng(seed) if fix_seed else None
        if w_star == None:
            w_star = 1*torch.randn(dimension, dtype=FEATURE_TYPE,
                                   generator=generator)
        assert w_star.size() == torch.Size([dimension])
        self.w_star = w_star
        self.noise = noise
        X = torch.randn((set_size, dimension), dtype=FEATURE_TYPE,
                        generator=generator)
        X.add_(torch.rand(1, generator=generator))
        # X.add_(torch.rand(dimension))
        # for i in range(set_size):
        #     X[i].div_(X[i].norm())
        Y = torch.matmul(X, w_star)
        Y.add_(torch.randn(Y.shape, dtype=VALUE_TYPE, generator=generator),
               alpha=noise)
        name = f'ToySet_D={dimension}_N={set_size}'
        super().__init__(name=name, features=X, targets=Y)

class LeastSquareToyTask(Task):
    def __init__(self):
        model = None
        
        # dimension = 100
        # data_cnt = 10
        dimension = 200
        data_cnt = 50
        
        dataset = LeastSquareToySet(data_cnt, dimension, 
                                    # noise=0, 
                                    fix_seed=True)
        
        super_params = {
            'rounds': 20,
            'display_interval': 1000,
            
            'primal_weight_decay': 5e-3,
            'dual_weight_decay': 1e-3,
            'penalty': 6e-2,
            # 'penalty': 1.37e-1, # minimum penalty
            
            'lr': 3e-2,
        }
        get_train_iter = train_full_batch_generator
        get_val_iter = partial(val_full_batch_generator, dataset=dataset)
        loss_fn = least_square_loss
        super().__init__(weight_decay=0, dataset=dataset, model=model,
                         loss_fn=loss_fn, 
                         get_train_iter=get_train_iter,
                         get_val_iter=get_val_iter,
                         initialize_fn=RandomInitialize(),
                         super_params=super_params,
                         name=f'LS_{dataset.name}', model_name='LinearModel')
        

