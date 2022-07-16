import random

import torch

from ByrdLab.library.dataset import DistributedDataSets
from ByrdLab.library.partition import TrivalPartition
from ByrdLab.library.learnRateController import constant_lr
from ByrdLab.library.RandomNumberGenerator import random_rng, torch_rng

class IterativeEnvironment():
    def __init__(self, name, lr, lr_ctrl=None,
                 rounds=10, display_interval=1000, total_iterations=None,
                 seed=None, fix_seed=False,
                 *args, **kw):
        '''
        display_interval: algorithm record the running information (e.g. loss,
                         accuracy) every 'display_interval' iterations
        rounds: totally 'rounds' information data will be records
        total_iterations: total iterations. We need to specify at least two of
                            arguments 'display_interval', 'rounds' and 
                            'total_iterations'.
                            'total_iterations' has to satisfy 
                            display_interval * rounds = total_iterations
        '''
        
        assert not fix_seed or seed != None
    
        # algorithm information
        self.name = name
        self.lr = lr
        self.seed = seed
        self.fix_seed = fix_seed
        
        # determine the runing time
        if rounds == None:
            rounds = total_iterations // display_interval
        elif display_interval == None:
            display_interval = total_iterations // rounds
        elif total_iterations == None:
            total_iterations = display_interval * rounds
        else:
            assert total_iterations == display_interval * rounds
        self.rounds = rounds
        self.display_interval = display_interval
        self.total_iterations = total_iterations
        
        # random number generator
        self.rng = random
        self.torch_rng = torch.default_generator
        
        # learning rate controller
        if lr_ctrl is None:
            self.lr_ctrl = constant_lr()
        else:
            self.lr_ctrl = lr_ctrl
        self.lr_ctrl.set_init_lr(lr)
            
    def construct_rng(self):
        # construct random number generator
        if self.fix_seed:
            self.rng = random_rng(self.seed)
            self.torch_rng = torch_rng(self.seed)
            
    def lr_path(self):
        return [self.lr_ctrl.get_lr(r * self.display_interval) 
                for r in range(self.rounds)]
        
    def set_params_suffix(self, params_show_names):
        # add suffix of parameters that is being tuned like: 
        # SGD_lr_0.01_momentum_0.9
        for params_code_name in params_show_names:
            params_value = self.__getattribute__(params_code_name)
            params_show_name = params_show_names[params_code_name]
            if params_show_name != '':
                self.name += f'_{params_show_name}={params_value}'
            else:
                self.name += f'_{params_value}'
            

class ByzantineEnvironment(IterativeEnvironment):
    def __init__(self, name, lr, model, weight_decay, 
                 dataset, loss_fn, initialize_fn=None, lr_ctrl=None,
                 get_train_iter=None, get_val_iter=None, 
                 partition_cls=TrivalPartition, 
                 honest_size=-1, byzantine_size=-1, 
                 honest_nodes=None, byzantine_nodes=None, attack=None,
                 rounds=10, display_interval=1000, total_iterations=None,
                 seed=None, fix_seed=False,
                 *args, **kw):
        super().__init__(name, lr, lr_ctrl, rounds, display_interval,
                         total_iterations, seed, fix_seed)

        # ====== check validity ======
        assert (honest_nodes is not None and honest_size < 0) \
            or (honest_nodes is not None and len(honest_nodes) == honest_size > 0) \
            or (honest_nodes is None and honest_size > 0)
        assert (byzantine_nodes is not None and len(byzantine_nodes) == byzantine_size >= 0) \
            or (byzantine_nodes is not None and byzantine_size < 0) \
            or (byzantine_nodes is None and byzantine_size >= 0) \
        
        # ====== define node set ======
        if honest_nodes is None:
            self.honest_nodes = list(range(honest_size))
            self.honest_size = honest_size
        else:
            self.honest_nodes = sorted(honest_nodes)
            self.honest_size = len(self.honest_nodes)
        if byzantine_nodes is None:
            self.byzantine_nodes = list(range(honest_size, honest_size+byzantine_size))
            self.byzantine_size = byzantine_size
        else:
            self.byzantine_nodes = sorted(byzantine_nodes)
            self.byzantine_size = len(byzantine_nodes)
        self.nodes = sorted(self.honest_nodes + self.byzantine_nodes)
        self.node_size = self.honest_size + self.byzantine_size
        
        # ====== define properties ======
        assert self.byzantine_size == 0 or attack != None
        self.attack = attack
        self.model = model
        self.initialize_fn = initialize_fn
        
        # ====== task information ======
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.get_train_iter = get_train_iter
        self.get_val_iter = get_val_iter

        # distribute dataset
        self.dataset = dataset
        dist_dataset = DistributedDataSets(dataset=dataset, 
                                           partition_cls=partition_cls,
                                           nodes=self.nodes,
                                           honest_nodes=self.honest_nodes)
        self.partition_name = dist_dataset.partition.name
        self.dist_dataset = dist_dataset
        
    def run(self, *args, **kw):
        raise NotImplementedError


class DecentralizedByzantineEnvironment(ByzantineEnvironment):
    def __init__(self, graph, *args, **kw):
        super(DecentralizedByzantineEnvironment, self).__init__(
            honest_nodes=graph.honest_nodes,
            byzantine_nodes=graph.byzantine_nodes,
            *args, **kw
        )
        self.graph = graph
        