
from ByrdLab.environment import Byz_Env
import torch

from ByrdLab import FEATURE_TYPE
from .tasks.logisticRegression import logistic_regression, logistic_regression_loss
from .tasks.logisticRegression import accuracy, get_varience
from .library.sample import local_sample, local_batch_sample, local_dataset_iterator
from .library.tool import log
from .library.RandomNumberGenerator import random_rng, torch_rng

class SGD(Byz_Env):
    def __init__(self, *args, **kw):
        super().__init__(name='SGD', *args, **kw)
    def run(self, w0):
        self.construct_rng_pack()
        # initialize
        w = w0.clone().detach()
        path = [logistic_regression_loss(w, self.train_set, self.weight_decay)]
        variencePath = []
        log('[SGD] initial loss={:.6f}, accuracy={:.2f} lr={:}'.format(
            path[0], accuracy(w, self.train_set), self.lr
        ))
        
        # 中间变量分配空间
        new_G = torch.zeros_like(w0, dtype=FEATURE_TYPE)
        message = torch.zeros(self.node_size, len(w0), dtype=FEATURE_TYPE)

        log('[begin optimizing]')
        for r in range(self.rounds):
            for k in range(self.display_interval):
                # 诚实节点更新
                for node in range(self.honest_size):
                    x, y = self.rng_pack.random.choice(self.dist_train_set[node])
                    # 更新梯度表
                    predict = logistic_regression(w, x)
                    err = (predict-y).data
                    new_G[:-1] = err*x
                    new_G[-1] = err
                    new_G.add_(w, alpha=self.weight_decay)
                    
                    gradient = new_G
                    
                    message[node].copy_(gradient.data)

                # 同步
                # Byzantine攻击
                if self.attack != None:
                    self.attack(message, self.byzantine_size)
                g = self.aggregation(message)
                w.add_(g.data, alpha=-self.lr)
                
            loss = logistic_regression_loss(w, self.train_set, self.weight_decay)
            acc = accuracy(w, self.train_set)
            path.append(loss)
            var = get_varience(message, self.honest_size)
            variencePath.append(var)
            log('[SGD] {}/{} rounds (interval: {:.0f}), loss={:.9f}, accuracy={:.2f}, var={:.9f}'.format(
                r+1, self.rounds, self.display_interval, loss, acc, var
            ))
        return w, path, variencePath

class BatchSGD(Byz_Env):
    def __init__(self, batch_size=50, *args, **kw):
        super().__init__(name='BatchSGD', *args, **kw)
        self.batch_size = batch_size
    def run(self, w0):
        self.construct_rng_pack()
        # initial化
        w = w0.clone().detach()

        path = [logistic_regression_loss(w, self.train_set, self.weight_decay)]
        variencePath = []
        log('[BatchSGD] initial loss={:.6f}, accuracy={:.2f} lr={:}'.format(
            path[0], accuracy(w, self.train_set), self.lr)
        )
        
        # 中间变量分配空间
        new_G = torch.zeros_like(w0, dtype=FEATURE_TYPE)
        message = torch.zeros(self.node_size, len(w0), dtype=FEATURE_TYPE)

        log('[begin optimizing]')
        for r in range(self.rounds):
            for k in range(self.display_interval):
                # 诚实节点更新
                for node in range(self.honest_size):
                    gradient = torch.zeros_like(new_G)
                    
                    for _ in range(self.batch_size):
                        x, y = self.rng_pack.random.choice(self.dist_train_set[node])
                        # 更新梯度表
                        predict = logistic_regression(w, x)
                        err = (predict-y).data
                        new_G[:-1] = err*x
                        new_G[-1] = err
                        new_G.add_(w, alpha=self.weight_decay)
                        gradient.add_(new_G, alpha=1/self.batch_size)
                    message[node].copy_(gradient.data)

                # 同步
                # Byzantine攻击
                if self.attack != None:
                    self.attack(message, self.byzantine_size)
                g = self.aggregation(message)
                w.add_(g.data, alpha=-self.lr)
                
            loss = logistic_regression_loss(w, self.train_set, self.weight_decay)
            acc = accuracy(w, self.train_set)
            path.append(loss)
            var = get_varience(message, self.honest_size)
            variencePath.append(var)
            log('[BatchSGD] {}/{} rounds (interval: {:.0f}), loss={:.9f}, accuracy={:.2f}, var={:.9f}'.format(
                r+1, self.rounds, self.display_interval, loss, acc, var
            ))
        return w, path, variencePath

class SAGA(Byz_Env):
    def __init__(self, *args, **kw):
        super().__init__(name='SAGA', *args, **kw)
    def run(self, w0):
        self.construct_rng_pack()
        # initialize
        w = w0.clone().detach()

        store = torch.zeros([len(self.train_set), w.size(0)],
                            requires_grad=False, dtype=FEATURE_TYPE)
        for index in range(len(self.train_set)):
            x, y = self.train_set[index]
            predict = logistic_regression(w, x)

            err = (predict-y).data
            store[index][:-1] = err*x
            store[index][-1] = err
            store[index].add_(w, alpha=self.weight_decay)

        G_avg = torch.stack([
            store[self.partition[i]:self.partition[i+1]].mean(dim=0)
            for i in range(self.honest_size)
        ])
        path = [logistic_regression_loss(w, self.train_set, self.weight_decay)]
        variencePath = []
        log('[SAGA] initial loss={:.6f}, accuracy={:.2f} lr={:}'.format(
            path[0], accuracy(w, self.train_set), self.lr
        ))
        
        # 中间变量分配空间
        new_G = torch.zeros_like(w0, dtype=FEATURE_TYPE)
        message = torch.zeros(self.node_size, len(w0), dtype=FEATURE_TYPE)

        log('[begin optimizing]')
        for r in range(self.rounds):
            for k in range(self.display_interval):
                # 诚实节点更新
                for node in range(self.honest_size):
                    x, y = self.rng_pack.random.choice(self.dist_train_set[node])
                    # 更新梯度表
                    predict = logistic_regression(w, x)

                    old_G = store[index]
                    err = (predict-y).data
                    new_G[:-1] = err*x
                    new_G[-1] = err
                    new_G.add_(w, alpha=self.weight_decay)

                    gradient = new_G.data - old_G.data + G_avg[node].data

                    G_avg[node].add_(new_G.data - old_G.data,
                                    alpha=1 / self.data_per_node[node])
                    store[index] = new_G.data

                    message[node].copy_(gradient.data)

                # 同步
                # Byzantine攻击
                if self.attack != None:
                    self.attack(message, self.byzantine_size)
                g = self.aggregation(message)
                w.add_(g.data, alpha=-self.lr)
                
            loss = logistic_regression_loss(w, self.train_set, self.weight_decay)
            acc = accuracy(w, self.train_set)
            path.append(loss)
            var = get_varience(message, self.honest_size)
            variencePath.append(var)
            log('[SAGA] {}/{} rounds (interval: {:.0f}), loss={:.9f}, accuracy={:.2f}, var={:.9f}'.format(
                r+1, self.rounds, self.display_interval, loss, acc, var
            ))
        return w, path, variencePath

class SVRG(Byz_Env):
    def __init__(self, snapshotInterval=6000, *args, **kw):
        super().__init__(name='SVRG', *args, **kw)
        self.snapshotInterval = snapshotInterval
    def run(self, w0):
        self.construct_rng_pack()
        # initial化
        w = w0.clone().detach()

        snapshot_g = torch.zeros(self.honest_size, len(w0), dtype=FEATURE_TYPE)
        snapshot_w = torch.zeros(len(w0), dtype=FEATURE_TYPE)

        path = [logistic_regression_loss(w, self.train_set, self.weight_decay)]
        variencePath = []
        log('[SVRG] initial loss={:.6f}, accuracy={:.2f} lr={:}'.format(
            path[0], accuracy(w, self.train_set), self.lr)
        )
        
        # 中间变量分配空间
        message = torch.zeros(self.node_size, len(w0), dtype=FEATURE_TYPE)

        log('[begin optimizing]')
        for r in range(self.rounds):
            for k in range(self.display_interval):
                # snapshot
                if (r*self.display_interval + k) % self.snapshotInterval == 0:
                    snapshot_g.zero_()
                    for node in range(self.honest_size):
                        for x, y in self.dist_train_set[node]:
                            # 更新梯度表
                            predict = logistic_regression(w, x)

                            err = (predict-y).data
                            snapshot_g[node][:-1].add_(err*x, alpha=1/self.data_per_node[node])
                            snapshot_g[node][-1].add_(err, alpha=1/self.data_per_node[node])
                        snapshot_g[node].add_(w, alpha=self.weight_decay)
                    snapshot_w.copy_(w)
                
                # 诚实节点更新
                message.zero_()
                for node in range(self.honest_size):
                    x, y = self.rng_pack.random.choice(self.dist_train_set[node])
                    # 随机梯度
                    predict = logistic_regression(w, x)
                    err = (predict-y).data
                    message[node][:-1].add_(x, alpha=err)
                    message[node][-1].add_(1, alpha=err)
                    message[node].add_(w, alpha=self.weight_decay)
                    
                    # 修正梯度
                    predict = logistic_regression(snapshot_w, x)
                    err = (predict-y).data
                    message[node][:-1].add_(x, alpha=-err)
                    message[node][-1].add_(1, alpha=-err)
                    message[node].add_(snapshot_w, alpha=-self.weight_decay)
                    
                    message[node].add_(snapshot_g[node], alpha=1)
                    
                # 同步
                # Byzantine攻击
                if self.attack != None:
                    self.attack(message, self.byzantine_size)
                g = self.aggregation(message)
                w.add_(g, alpha=-self.lr)
                
            loss = logistic_regression_loss(w, self.train_set, self.weight_decay)
            acc = accuracy(w, self.train_set)
            path.append(loss)
            var = get_varience(message, self.honest_size)
            variencePath.append(var)
            log('[SVRG] {}/{} rounds (interval: {:.0f}), loss={:.9f}, accuracy={:.2f}, var={:.9f}'.format(
                r+1, self.rounds, self.display_interval, loss, acc, var
            ))
        return w, path, variencePath

class SARAH(Byz_Env):
    def __init__(self, snapshotInterval=6000, *args, **kw):
        super().__init__(name='SARAH', *args, **kw)
        self.snapshotInterval = snapshotInterval
    def run(self, w0):
        self.construct_rng_pack()
        # initilize
        w = w0.clone().detach()

        lastGradients = torch.zeros_like(w0, dtype=FEATURE_TYPE)

        path = [logistic_regression_loss(w, self.train_set, self.weight_decay)]
        variencePath = []
        log('[SARAH] initial loss={:.6f}, accuracy={:.2f} lr={:}'.format(
            path[0], accuracy(w, self.train_set), self.lr)
        )
        
        # 中间变量分配空间
        message = torch.zeros(self.node_size, len(w0), dtype=FEATURE_TYPE)
        newG = torch.zeros_like(w0, dtype=FEATURE_TYPE)
        lastw = torch.zeros_like(w0, dtype=FEATURE_TYPE)
        
        # 随机的停止期限
        randomStop = 1

        log('[begin optimizing]')
        for r in range(self.rounds):
            for k in range(self.display_interval):
                # snapshot
                if (r*self.display_interval + k) % randomStop == 0:
                    message.zero_()
                    for node in range(self.honest_size):
                        for x, y in self.dist_train_set[node]:
                            predict = logistic_regression(w, x)

                            err = (predict-y).data
                            message[node][:-1].add_(
                                err*x, alpha=1/self.data_per_node[node])
                            message[node][-1].add_(
                                err, alpha=1/self.data_per_node[node])
                        message[node].add_(w, alpha=self.weight_decay)
                    
                    # 首次更新
                    if self.attack != None:
                        self.attack(message, self.byzantine_size)
                    g = self.aggregation(message)
                    lastw.copy_(w)
                    w.add_(g, alpha=-self.lr)
                    # 指定下一次停止时间
                    randomStop = self.rng_pack.random.randint(
                        1, self.snapshotInterval-1)
                
                # 诚实节点更新
                for node in range(self.honest_size):
                    x, y = self.random.rng.random.choice(self.dist_train_set[node])
                    # 随机梯度
                    predict = logistic_regression(w, x)
                    err = (predict-y).data
                    message[node][:-1].add_(x, alpha=err)
                    message[node][-1].add_(1, alpha=err)
                    message[node].add_(w, alpha=self.weight_decay)
                    
                    # 修正梯度
                    predict = logistic_regression(lastw, x)
                    err = (predict-y).data
                    message[node][:-1].add_(x, alpha=-err)
                    message[node][-1].add_(1, alpha=-err)
                    message[node].add_(lastw, alpha=-self.weight_decay)

                # 保存旧结果
                lastw.copy_(w)
                # 同步, Byzantine攻击
                if self.attack != None:
                    self.attack(message, self.byzantine_size)
                g = self.aggregation(message)
                w.add_(g, alpha=-self.lr)
                
            loss = logistic_regression_loss(w, self.train_set, self.weight_decay)
            acc = accuracy(w, self.train_set)
            path.append(loss)
            var = get_varience(message, self.honest_size)
            variencePath.append(var)
            log('[SARAH] {}/{} rounds (interval: {:.0f}), loss={:.9f}, accuracy={:.2f}, var={:.9f}'.format(
                r+1, self.rounds, self.display_interval, loss, acc, var
            ))
        return w, path, variencePath

class ByrD2SAGA(Byz_Env):
    def __init__(self, *args, **kw):
        super().__init__(name='ByrD2SAGA', *args, **kw)
    def run(self, w0):
        self.construct_rng_pack()
        # initialize
        w = w0.clone().detach()

        store = torch.zeros([len(self.train_set), w.size(0)],
            requires_grad=False, dtype=FEATURE_TYPE)
        for index in range(len(self.train_set)):
            x, y = self.train_set[index]
            predict = logistic_regression(w, x)

            err = (predict-y).data
            store[index][:-1] = err*x
            store[index][-1] = err
            store[index].add_(w, alpha=self.weight_decay)

        G_avg = torch.stack([
            store[self.partition[i][0]:self.partition[i][1]].mean(dim=0)
                for i in range(self.honest_size)
        ])
        path = [logistic_regression_loss(w, self.train_set, self.weight_decay)]
        variencePath = []
        log('[SAGA] initial loss={:.6f}, accuracy={:.2f} lr={:}'.format(
            path[0], accuracy(w, self.train_set), self.lr))
        
        # 新梯度存储空间
        new_G = torch.zeros_like(w0, dtype=FEATURE_TYPE)
        # 节点传输的信息
        message = torch.zeros(self.node_size, len(w0), dtype=FEATURE_TYPE)
        # 聚合后梯度
        g = torch.zeros_like(w0, dtype=FEATURE_TYPE)

        log('[begin optimizing]')
        for r in range(self.rounds):
            for k in range(self.display_interval):
                # 诚实节点更新
                message[:self.honest_size].mul_(-1)
                message[:self.honest_size].add_(g)
                for node in range(self.honest_size):
                    x, y = self.rng_pack.choice(self.dist_train_set[node])
                    # 更新梯度表
                    predict = logistic_regression(w, x)

                    old_G = store[index]
                    err = (predict-y).data
                    new_G[:-1] = err*x
                    new_G[-1] = err
                    new_G.add_(w, alpha=self.weight_decay)

                    gradient = new_G.data - old_G.data + G_avg[node].data

                    G_avg[node].add_(new_G.data - old_G.data,
                                        alpha=1 / self.data_per_node[node])
                    store[index] = new_G.data

                    message[node].add_(gradient.data)
                print(get_varience(message, self.honest_size))
                # 同步
                # Byzantine攻击
                if self.attack != None:
                    self.attack(message, self.byzantine_size)
                g = self.aggregation(message)
                w.add_(g.data, alpha=-self.lr)
                
            loss = logistic_regression_loss(w, self.train_set, self.weight_decay)
            acc = accuracy(w, self.train_set)
            path.append(loss)
            var = get_varience(message, self.honest_size)
            variencePath.append(var)
            log('[SAGA] {}/{} rounds (interval: {:.0f}), loss={:.9f}, accuracy={:.2f}, var={:.9f}'.format(
                r+1, self.rounds, self.display_interval, loss, acc, var
            ))
        return w, path, variencePath
