from ByrdLab.library.dataset import DataPackage, StackedDataSet
from ByrdLab import FEATURE_TYPE, VALUE_TYPE
from ByrdLab.library.RandomNumberGenerator import RngPackage, torch_rng
from functools import partial

import torch

from ByrdLab.tasks import Task
from ByrdLab.library.initialize import RandomInitialize

class LeastSqaure_LinearModel(torch.nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=feature_dimension,
                                      out_features=1, bias=False)
    def forward(self, features):
        features = features.view(features.size(0), -1)
        return self.linear(features)

# def train_full_batch_generator(dist_dataset, node,
#                                rng_pack: RngPackage=RngPackage()):
#     while True:
#         yield dist_dataset[node][:]

def train_full_batch_generator(dataset,
                               rng_pack: RngPackage=RngPackage()):
    while True:
        yield dataset[:]
        
        
def test_full_batch_generator(dataset):
    yield dataset[:]
        
def least_square_loss(predictions, targets):
    predictions = predictions.squeeze(dim=-1)
    return ((predictions-targets)**2).mean()

# def least_square_loss(predictions, targets):
#     return ((predictions-targets)**2).sum()


class LeastSquareToySet(DataPackage):
    def __init__(self, set_size, dimension, noise=0.1, w_star=None,
                 fix_seed=False, seed=20):
        self.size_set = set_size
        self.dimension = dimension
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
        dataset = StackedDataSet(features=X, targets=Y)
        super().__init__(name=name, train_set=dataset, test_set=dataset)

class LeastSquareToyTask(Task):
    def __init__(self, data_package):
        # model = LeastSqaure_LinearModel
        
        # dimension = 100
        # data_cnt = 10
        # dimension = 200
        # data_cnt = 50
        
        # data_package = LeastSquareToySet(data_cnt, dimension, 
        #                                  # noise=0, 
        #                                  fix_seed=True)
        self.data_package = data_package
        model = LeastSqaure_LinearModel(data_package.dimension)
        
        super_params = {
            'rounds': 100,
            'display_interval': 1,
            
            'primal_weight_decay': 5e-3,
            'dual_weight_decay': 1e-3,
            # 'penalty': 6e-2,
            # 'penalty': 1.37e-1, # minimum penalty
            
            'lr': 3e-2,
        }
        get_train_iter = train_full_batch_generator
        get_test_iter = partial(test_full_batch_generator, 
                                dataset=self.data_package.test_set)
        loss_fn = least_square_loss
        super().__init__(weight_decay=0, data_package=self.data_package, model=model,
                         loss_fn=loss_fn, 
                         test_fn=None,
                         get_train_iter=get_train_iter,
                         get_test_iter=get_test_iter,
                         initialize_fn=RandomInitialize(),
                         super_params=super_params,
                         name=f'LS_{data_package.name}', model_name='LinearModel')
        

