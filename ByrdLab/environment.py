from ByrdLab.DistributedModule import DistributedModule
from ByrdLab.library.dataset import DataPackage, DistributedDataSets, DistributedDataSets_over_honest_and_byz_nodes
from ByrdLab.library.partition import Partition, TrivalPartition
from ByrdLab.library.learnRateController import constant_lr
from ByrdLab.library.RandomNumberGenerator import RngPackage

class AlgorithmEnvironment():
    def __init__(self, name, *args, **kw) -> None:
        self.name = name
        
    def run(self, *args, **kw):
        raise NotImplementedError


# Iterative Environment
class IterativeEnvironment(AlgorithmEnvironment):
    '''
    A base class for any algorithm requiring iteration
    '''
    def __init__(self, name, 
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
        super().__init__(name=name, *args, **kw)
        
        assert not fix_seed or seed != None
    
        # algorithm information
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
        self.rng_pack = RngPackage(seed=None)
            
    def construct_rng_pack(self):
        # construct random number generator
        if self.fix_seed:
            self.rng_pack.set_seed(self.seed)
            
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
            
            
# Optimization Environment
class Opt_Env(IterativeEnvironment):
    def __init__(self, name, lr, model, weight_decay, 
                 data_package: DataPackage, 
                 loss_fn, test_fn, initialize_fn=None, lr_ctrl=None,
                 get_train_iter=None, get_test_iter=None, 
                 rounds=10, display_interval=1000, total_iterations=None,
                 seed=None, fix_seed=False,
                 *args, **kw):
        super().__init__(name=name, rounds=rounds, 
                         display_interval=display_interval,
                         total_iterations=total_iterations, 
                         seed=seed, fix_seed=fix_seed,
                         *args, **kw)
        
        # ====== define properties ======
        self.model = model
        self.initialize_fn = initialize_fn
        
        # ====== learning rate controller ======
        self.lr = lr
        if lr_ctrl is None:
            self.lr_ctrl = constant_lr()
        else:
            self.lr_ctrl = lr_ctrl
        self.lr_ctrl.set_init_lr(lr)
        
        # ====== task information ======
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.test_fn = test_fn
        self.get_train_iter = get_train_iter
        self.get_test_iter = get_test_iter
        self.data_package = data_package


# Distributed Environment in the present of Byzantine node
class Byz_Env(AlgorithmEnvironment):
    def __init__(self, honest_size=-1, byzantine_size=-1, 
                 honest_nodes=None, byzantine_nodes=None, attack=None,
                 *args, **kw):
        super().__init__(*args, **kw)

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

        # Byzantine nodes can act like honest nodes 
        assert self.byzantine_size == 0 or attack != None
        self.attack = attack


class Dist_Model_Env(Byz_Env):
    def __init__(self, *args, **kw): 
        super().__init__(*args, **kw)
        
    def construct_dist_models(self, model, node_size):
        return DistributedModule(model, node_size)
        
    def initilize_models(self, dist_models, consensus=False):
        if self.initialize_fn is None:
            return
        # for node in self.honest_nodes:
        for node in self.nodes:
            dist_models.activate_model(node)
            model = dist_models.model
            seed = self.seed if consensus else node + self.seed
            self.initialize_fn(model, fix_init_model=self.fix_seed, seed=seed)


class Dist_Dataset_Opt_Env(Byz_Env, Opt_Env):
    def __init__(self, partition_cls: Partition=TrivalPartition, 
                 *args, **kw): 
        super().__init__(*args, **kw)
        
        train_set = self.data_package.train_set
        # # ====== distribute dataset ======
        # dist_train_set = DistributedDataSets(dataset=train_set, 
        #                                      partition_cls=partition_cls,
        #                                      nodes=self.nodes,
        #                                      honest_nodes=self.honest_nodes,
        #                                      rng_pack=self.rng_pack)

        # ====== distribute dataset over honest and byzantine nodes ======
        dist_train_set = DistributedDataSets_over_honest_and_byz_nodes(dataset=train_set, 
                                                                       partition_cls=partition_cls,
                                                                       nodes=self.nodes,
                                                                       rng_pack=self.rng_pack)
        self.partition_name = dist_train_set.partition.name
        self.dist_train_set = dist_train_set


# Decentralized Optimization Environment in the presence of Byzantine nodes
class Dec_Byz_Opt_Env(
    Dist_Model_Env, Dist_Dataset_Opt_Env
):
    def __init__(self, graph, *args, **kw):
        super().__init__(
            honest_nodes=graph.honest_nodes,
            byzantine_nodes=graph.byzantine_nodes,
            *args, **kw
        )
        self.graph = graph
        
        
# Decentralized Iterative Environment in the presence of Byzantine nodes
class Dec_Byz_Iter_Env(
    IterativeEnvironment, Dist_Model_Env
):
    def __init__(self, graph, *args, **kw):
        super().__init__(
            honest_nodes=graph.honest_nodes,
            byzantine_nodes=graph.byzantine_nodes,
            *args, **kw
        )
        self.graph = graph
    