import math
import bisect

class learningRateController():
    def __init__(self, name):
        self.name = name
    def set_init_lr(self, init_lr):
        self.init_lr = init_lr
    def get_lr(self, iteration):
        pass
    
class constant_lr(learningRateController):
    def __init__(self):
        super().__init__(name='constLR')
    def get_lr(self, iteration):
        return self.init_lr
    
class one_over_sqrt_k_lr(learningRateController):
    '''
    O(1/sqrt(k)) decreasing step size
    we choose proper constant so that variable 'decreasing_factor'
    is 1 at iteration 0 and 'final_proportion' at iteration 'total_iteration'
    '''
    def __init__(self, total_iteration=5000, final_proportion=1/10,
                 a=None, b=None):
        # we choose proper constant so that 
        # variable 'decreasing_factor' is 1 at iteration 1 and
        # 1/final_proportion at iteration 'total_iteration'
        super().__init__(name='invSqrtLR')
        if a == None or b == None:
            b = (total_iteration * final_proportion**2) \
                / (1 - final_proportion**2)
            a = math.sqrt(b)
        self.a = a
        self.b = b
    def get_lr(self, iteration):
        # a / sqrt(k+b) learning rate
        decreasing_factor = self.a / math.sqrt(iteration + self.b)
        return self.init_lr * decreasing_factor
    
class one_over_k_lr(learningRateController):
    '''
    O(1/k) decreasing step size
    we choose proper constant so that variable 'decreasing_factor'
    is 1 at iteration 1 and 1/10 at iteration 'total_iteration'
    '''
    def __init__(self, total_iteration=5000, final_proportion=1/10,
                 a=None, b=None):
        # we choose proper constant so that 
        # variable 'decreasing_factor' is 1 at iteration 1 and
        # 1/final_proportion at iteration 'total_iteration'
        super().__init__(name='invLR')
        if a == None or b == None:
            b = (total_iteration * final_proportion - 1) \
             / (1 - final_proportion)
            a = 1 + b
        self.a = a
        self.b = b
            
    def get_lr(self, iteration):
        # a / sqrt(k+b) learning rate
        decreasing_factor = self.a / (iteration + self.b)
        return self.init_lr * decreasing_factor
    
class ladder_lr(learningRateController):
    def __init__(self, decreasing_iter_ls=[], proportion_ls=[]):
        assert len(decreasing_iter_ls) == len(proportion_ls)
        super().__init__(name='ladderLR')
        self.decreasing_iter_ls = decreasing_iter_ls.copy()
        self.proportion_ls = proportion_ls.copy()
        if len(self.decreasing_iter_ls) == 0 or self.decreasing_iter_ls[0] != 0:
            self.decreasing_iter_ls.insert(0, 0)
            self.proportion_ls.insert(0, 1)
    def get_lr(self, iteration):
        pos = bisect.bisect_right(self.decreasing_iter_ls, iteration)
        return self.proportion_ls[pos - 1] * self.init_lr
        