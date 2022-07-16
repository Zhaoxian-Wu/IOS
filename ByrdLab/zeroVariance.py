'''
The algorithms in this file are totally the same of that in file 
"distributed Optimizer.py", only difference is that the algorithms 
here load the same dataset across all nodes, causing the outter
variation zero exactly
'''
import random

import torch

from lib.tool import log
from lib.loss import logistic_regression, logistic_regression_loss
from lib.loss import accuracy, get_varience
from distributedOptimizer import SGD, BatchSGD, SAGA


class SGD_ZV(SGD):
    def __init__(self, *args, **kw):
        super(SGD_ZV, self).__init__(name='SGD_ZV', *args, **kw)
        self.partition = [[0, len(self.dataset)] 
                            for _ in range(self.honest_size)]
        self.data_per_node = [len(self.dataset)] * self.honest_size

class BatchSGD_ZV(BatchSGD):
    def __init__(self, *args, **kw):
        super(BatchSGD_ZV, self).__init__(name='BatchSGD_ZV', *args, **kw)
        self.partition = [[0, len(self.dataset)] 
                            for _ in range(self.honest_size)]
        self.data_per_node = [len(self.dataset)] * self.honest_size

class SAGA_ZV(SAGA):
    def __init__(self, *args, **kw):
        super(SAGA_ZV, self).__init__(name='SAGA_ZV', *args, **kw)
        self.partition = [[0, len(self.dataset)] 
                            for _ in range(self.honest_size)]
        self.data_per_node = [len(self.dataset)] * self.honest_size
