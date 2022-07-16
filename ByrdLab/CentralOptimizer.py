import random

import torch

from lib.loss import logistic_regression, logistic_regression_loss
from lib.loss import accuracy
from lib.tool import log
from lib.config import FEATURE_TYPE

def CentralSAGA(w0, gamma, weight_decay, dataset, rounds=10,
        displayInterval=1000, SEED=100, fixSeed=False, **kw):

    # 初始化
    w = w0.clone().detach()

    set_size = len(dataset)
    
    store = torch.zeros([set_size, w.size(0)], 
                requires_grad=False, dtype=FEATURE_TYPE)
    for index in range(set_size):
        x, y = dataset[index]
        predict = logistic_regression(w, x)

        err = (predict-y).data
        store[index][:-1] = err*x
        store[index][-1] = err
        store[index].add_(w, alpha=weight_decay)

    G_avg = torch.mean(store, dim=0)
    path = [logistic_regression_loss(w, dataset, weight_decay)]
    
    # 中间变量分配空间
    new_G = torch.zeros(w0.size(), dtype=FEATURE_TYPE)
    
    log('[SAGA]initial loss={:.6f}, accuracy={:.2f} gamma={:}'.format(path[0], accuracy(w, dataset), gamma))
    log('[begin]')
    for r in range(rounds):
        for k in range(displayInterval):
            # 更新梯度表
            index = random.randint(0, set_size-1)

            x, y = dataset[index]
            predict = logistic_regression(w, x)
            
            # 计算梯度
            old_G = store[index]
            err = (predict-y).data
            new_G[:-1] = err*x
            new_G[-1] = err
            new_G.add_(w, alpha=weight_decay)
            
            gradient = new_G.data - old_G.data + G_avg.data
            
            G_avg.add_(new_G.data - old_G.data, alpha=1/set_size)
            store[index] = new_G.data
            w.data.add_(gradient.data, alpha=-gamma)
            
        loss = logistic_regression_loss(w, dataset, weight_decay)
        acc = accuracy(w, dataset)
        path.append(loss)
        log('[SAGA]finish {}/{} rounds (interval: {:.0f}), loss={:.9f}, accuracy={:.2f}'.format(
            r+1, rounds, displayInterval, loss, acc
        ))
    return w, path, []

def SAGA_min(w0, gamma, weight_decay, dataset, epoch=1, **kw):

    # 初始化
    w = w0.clone().detach()

    set_size = len(dataset)
    
    store = torch.zeros([set_size, w.size(0)], requires_grad=False, dtype=FEATURE_TYPE)
    for index in range(set_size):
        x, y = dataset[index]
        predict = logistic_regression(w, x)

        err = -(y-predict).data
        store[index][:-1] = err*x
        store[index][-1] = err
        store[index].add_(w, alpha=weight_decay)

    G_avg = torch.mean(store, dim=0)
    
    # 中间变量分配空间
    new_G = torch.zeros(w0.size(), dtype=FEATURE_TYPE)
    for e in range(epoch):
        for _ in range(set_size):
            # 更新梯度表
            index = random.randint(0, set_size-1)

            x, y = dataset[index]
            predict = logistic_regression(w, x)
            
            # 计算梯度
            old_G = store[index]
            err = -(y-predict).data
            new_G[:-1] = err*x
            new_G[-1] = err
            new_G.add_(w, alpha=weight_decay)
            
            gradient = new_G.data - old_G.data + G_avg.data
            
            G_avg.add_(new_G.data - old_G.data, alpha=1 / set_size)
            store[index] = new_G.data
            w.data.add_(gradient.data, alpha=-gamma)
        log('[SAGA] finish{:.0f}/{:.0f}'.format(e+1, epoch))
    
    return w

def CentralFinito(w0, gamma, weight_decay, dataset,
                    rounds=10, displayInterval=1000,
                    SEED=100, fixSeed=False, **kw):
    if fixSeed:
        random.seed(SEED)
        
    SET_SIZE = len(dataset)
    
    # 初始化
    w = w0.clone().detach()
    store = torch.stack([w]*SET_SIZE)

    # 构建梯度
    G_avg = torch.zeros_like(w)
    for index in range(SET_SIZE):
        x, y = dataset[index]
        # 更新梯度表
        predict = logistic_regression(w, x)

        err = (predict-y).data
        G_avg[:-1].add_(err*x, 1/SET_SIZE)
        G_avg[-1].add_(err, 1/SET_SIZE)
    
    # \bar{\phi}
    w_avg = store.mean(dim=0)
            
    path = [logistic_regression_loss(w, dataset, weight_decay)]
    log('[Finito]初始 loss={:.6f}, accuracy={:.2f} gamma={:}'.format(path[0], accuracy(w, dataset), gamma))
    
    # 中间变量分配空间
    new_G = torch.zeros_like(w0, dtype=torch.float64)

    log('开始迭代')
    for r in range(rounds):
        for k in range(displayInterval):
            # 更新模型
            w = w_avg - gamma * G_avg
            
            index = random.randint(0, SET_SIZE-1)
            
            #更新平均梯度
            x, y = dataset[index]
            # 减旧梯度
            predict = logistic_regression(store[index], x)
            err = (predict-y).data
            G_avg[:-1].sub_(err*x, alpha=1/SET_SIZE)
            G_avg[-1].sub_(err, alpha=1/SET_SIZE)
            # 加新梯度
            predict = logistic_regression(w, x)
            err = (predict-y).data
            G_avg[:-1].add_(err*x, alpha=1/SET_SIZE)
            G_avg[-1].add_(err, alpha=1/SET_SIZE)
            
            # 更新平均模型
            w_avg.add_(w, alpha=1/SET_SIZE)
            w_avg.sub_(store[index], alpha=1/SET_SIZE)
            store[index] = w
            
        loss = logistic_regression_loss(w, dataset, weight_decay)
        acc = accuracy(w, dataset)
        path.append(loss)
        log('[Finito]已迭代 {}/{} rounds (interval: {:.0f}), loss={:.9f}, accuracy={:.2f}'.format(
            r+1, rounds, displayInterval, loss, acc
        ))
    return w, path, []
