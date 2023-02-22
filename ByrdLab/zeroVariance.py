'''
The algorithms in this file are totally the same of that in file 
"distributed Optimizer.py", only difference is that the algorithms 
here load the same dataset across all nodes, causing the outter
variation zero exactly
'''
import random

import torch

from ByrdLab.library.tool import log
from ByrdLab.library.measurements import logistic_regression, logistic_regression_loss
from ByrdLab.library.measurements import accuracy, get_varience
from distributedOptimizer import SGD, BatchSGD, SAGA


class SGD_ZV(SGD):
    def __init__(self, *args, **kw):
        super().__init__(name='SGD_ZV', *args, **kw)
        self.partition = [[0, len(self.train_set)] 
                            for _ in range(self.honest_size)]
        self.data_per_node = [len(self.train_set)] * self.honest_size

class BatchSGD_ZV(BatchSGD):
    def __init__(self, *args, **kw):
        super().__init__(name='BatchSGD_ZV', *args, **kw)
        self.partition = [[0, len(self.train_set)] 
                            for _ in range(self.honest_size)]
        self.data_per_node = [len(self.train_set)] * self.honest_size

class SAGA_ZV(SAGA):
    def __init__(self, *args, **kw):
        super().__init__(name='SAGA_ZV', *args, **kw)
        self.partition = [[0, len(self.train_set)] 
                            for _ in range(self.honest_size)]
        self.data_per_node = [len(self.train_set)] * self.honest_size
