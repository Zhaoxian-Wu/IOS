import math
import random

import scipy.stats
import torch

from ByrdLab import FEATURE_TYPE
from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab.library.tool import MH_rule

def gaussian(messages, honest_nodes, byzantine_nodes, scale, torch_rng=None):
    # with the same mean and larger variance
    mu = torch.zeros(messages.size(1), dtype=FEATURE_TYPE)
    for node in honest_nodes:
        mu.add_(messages[node], alpha=1/len(honest_nodes))
    for node in byzantine_nodes:
        messages[node].copy_(mu)
        noise = torch.randn(messages.size(1), dtype=FEATURE_TYPE,
                            generator=torch_rng)
        messages[node].add_(noise, alpha=scale)
    
def sign_flipping(messages, honest_nodes, byzantine_nodes, scale,
                  noise_scale=0, torch_rng=None):
    mu = torch.zeros(messages.size(1), dtype=FEATURE_TYPE)
    for node in honest_nodes:
        mu.add_(messages[node], alpha=1/len(honest_nodes))
    melicious_message = -scale * mu
    for node in byzantine_nodes:
        noise = torch.randn(messages.size(1), dtype=FEATURE_TYPE,
                            generator=torch_rng)
        messages[node].copy_(melicious_message)
        messages[node].add_(noise, alpha=noise_scale)
             
def get_model_control(messages, honest_nodes, byzantine_nodes, target_message):
    s = torch.zeros(messages.size(1), dtype=FEATURE_TYPE)
    for node in honest_nodes:
        s.add_(messages[node])
    melicious_message = (target_message*len(honest_nodes)-s) / len(byzantine_nodes)
    return melicious_message

def get_model_control_weight(messages, honest_nodes, byzantine_nodes, target_message, weights):
    s = torch.zeros(messages.size(1), dtype=FEATURE_TYPE)
    for node in honest_nodes:
        s.add_(messages[node], alpha=weights[node])
    byzantine_weight = weights[byzantine_nodes].sum()
    melicious_message = (target_message-s) / byzantine_weight
    return melicious_message

def model_control(messages, honest_nodes, byzantine_nodes, target_message):
    melicious_message = get_model_control(messages, honest_nodes, 
                                          byzantine_nodes, target_message)
    for node in byzantine_nodes:
        messages[node].copy_(melicious_message)
    
def zero_attack(messages, honest_nodes, byzantine_nodes, noise_scale=0,
                torch_rng=None):
    target_message = torch.zeros(messages.size(1))
    melicious_message = get_model_control(messages, honest_nodes, 
                                          byzantine_nodes, target_message)
    for node in byzantine_nodes:
        messages[node].copy_(melicious_message)
        noise = torch.randn(messages.size(1), dtype=FEATURE_TYPE,
                            generator=torch_rng)
        messages[node].add_(noise, alpha=noise_scale)
        
def same_value_attack(messages, honest_nodes, byzantine_nodes, scale=1,
                      noise_scale=0, rng=None):
    c = 0
    for node in honest_nodes:
        # c += messages[node].mean().item()
        c += messages[node].mean().item() / len(honest_nodes)
    melicious_message = scale*c*torch.ones(messages.size(1))
    melicious_message /= math.sqrt(messages.size(1))
    for node in byzantine_nodes:
        messages[node].copy_(melicious_message)
        noise = torch.randn(messages.size(1), dtype=FEATURE_TYPE, generator=rng)
        messages[node].add_(noise, alpha=noise_scale)
    
    
class CentralizedAttack():
    def __init__(self, name, honest_nodes, byzantine_nodes):
        self.name = name
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
    
class CentralizedAttackWrapper(CentralizedAttack):
    def __init__(self, name, honest_nodes, byzantine_nodes, attack_fn, **kw):
        super().__init__(name=name, honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes)
        self.kw = kw
        self.attack_fn = attack_fn
    def run(self, messages):
        self.attack_fn(messages, self.honest_nodes, self.byzantine_nodes, **self.kw)
    
class C_gaussian(CentralizedAttackWrapper):
    def __init__(self, honest_nodes, byzantine_nodes, scale=30):
        super().__init__(name='gaussian', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes, 
                         attack_fn=gaussian, scale=scale)
        self.scale = scale
            
class C_sign_flipping(CentralizedAttackWrapper):
    def __init__(self, honest_nodes, byzantine_nodes, scale=3, noise_scale=0):
        super().__init__(name='sign_flipping', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes, 
                         attack_fn=sign_flipping, scale=scale,
                         noise_scale=noise_scale)
        self.scale = scale
        
class C_zero_gradient(CentralizedAttackWrapper):
    def __init__(self, honest_nodes, byzantine_nodes, noise_scale=0):
        super().__init__(name='zero_gradient', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes, 
                         attack_fn=zero_attack, noise_scale=noise_scale)
        
class C_isolation(CentralizedAttack):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='isolation', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes)
    def run(self, messages):
        melicious_message = get_model_control(messages, self.honest_nodes, 
                                              self.byzantine_nodes, 
                                              messages[-1])
        for node in self.byzantine_nodes:
            messages[node].copy_(melicious_message)

class C_same_value(CentralizedAttackWrapper):
    def __init__(self, honest_nodes, byzantine_nodes, scale=1, noise_scale=0):
        super().__init__(name='same_value', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes, scale=scale,
                         attack_fn=same_value_attack, noise_scale=noise_scale)

class decentralizedAttack():
    def __init__(self, name, graph):
        self.graph = graph
        self.name = name
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        raise NotImplementedError
    
class D_gaussian(decentralizedAttack):
    def __init__(self, graph, scale=30):
        super(D_gaussian, self).__init__(name='gaussian', graph=graph)
        self.scale = scale
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        honest_neighbors = self.graph.honest_neighbors[node]
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        mu = torch.mean(local_models[honest_neighbors], dim=0)
        for n in byzantine_neigbors:
            local_models[n].copy_(mu)
            noise = torch.randn(local_models.size(1), 
                                generator=rng_pack.torch,
                                dtype=FEATURE_TYPE)
            local_models[n].add_(noise, alpha=self.scale)
            
class D_sign_flipping(decentralizedAttack):
    def __init__(self, graph, scale=None):
        if scale is None:
            scale = 1
            name = 'sign_flipping'
        else:
            name = f'sign_flipping_s={scale}'
        super(D_sign_flipping, self).__init__(name=name, graph=graph)
        self.scale = scale
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        honest_neighbors = self.graph.honest_neighbors[node]
        byzantine_neigbor = self.graph.byzantine_neighbors[node]
        mu = torch.mean(local_models[honest_neighbors+[node]], dim=0)
        melicious_message = -self.scale * mu
        for n in byzantine_neigbor:
            local_models[n].copy_(melicious_message)
         
class D_zero_sum(decentralizedAttack):
    def __init__(self, graph):
        super(D_zero_sum, self).__init__(name='zero_sum', graph=graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        melicious_message = get_dec_model_control(self.graph, local_models, node, 
                                                  torch.zeros_like(local_models[node]))
        for n in byzantine_neigbors:
            local_models[n].copy_(melicious_message)
            
class D_zero_value(decentralizedAttack):
    def __init__(self, graph):
        super().__init__(name='zero_value', graph=graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        for n in byzantine_neigbors:
            local_models[n].copy_(torch.zeros_like(local_models[node]))
            
def get_dec_model_control(graph, messages, node, target_model):
    honest_neighbors = graph.honest_neighbors[node]
    byzantine_neigbors = graph.byzantine_neighbors[node]
    melicious_message = get_model_control(messages, honest_neighbors,
                                          byzantine_neigbors, target_model)
    return melicious_message

def get_dec_model_control_weight(graph, messages, node, target_model, weight):
    honest_neighbors = graph.honest_neighbors_and_itself[node]
    byzantine_neigbors = graph.byzantine_neighbors[node]
    melicious_message = get_model_control_weight(messages, honest_neighbors,
                                                 byzantine_neigbors,
                                                 target_model, weight)
    return melicious_message

class D_isolation(decentralizedAttack):
    def __init__(self, graph):
        super(D_isolation, self).__init__(name='isolation', graph=graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        melicious_message = get_dec_model_control(self.graph, local_models, node, 
                                                  local_models[node])
        for n in byzantine_neigbors:
            local_models[n].copy_(melicious_message)
            
class D_isolation_weight(decentralizedAttack):
    def __init__(self, graph):
        super().__init__(name='isolation_w', graph=graph)
        self.W = MH_rule(graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        melicious_message = get_dec_model_control_weight(self.graph, 
                                                         local_models, node, 
                                                         local_models[node],
                                                         self.W[node])
        for n in byzantine_neigbors:
            local_models[n].copy_(melicious_message)
        # avg = local_models[self.graph.neighbors_and_itself[node]].sum(dim=0) / (self.graph.neighbor_sizes[node]+1)

class D_sample_duplicate(decentralizedAttack):
    def __init__(self, graph):
        super().__init__(name='duplicate', graph=graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        honest_neighbors = self.graph.honest_neighbors[node]
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        duplicate_index = rng_pack.random.choice(honest_neighbors)
        for n in byzantine_neigbors:
            local_models[n].copy_(local_models[duplicate_index])
        
# A Little is Enough
class D_alie(decentralizedAttack):
    def __init__(self, graph, scale=None):
        if scale is None:
            name = 'alie'
        else:
            name = f'alie_scale={scale}'
        super().__init__(name=name, graph=graph)
        if scale is None:
            self.scale_table = [0] * self.graph.node_size
            for node in self.graph.honest_nodes:
                neighbors_size = self.graph.neighbor_sizes[node]
                byzantine_size = self.graph.byzantine_sizes[node]
                s = math.floor((neighbors_size+1)/2)-byzantine_size
                percent_point = (neighbors_size-s)/neighbors_size
                scale = scipy.stats.norm.ppf(percent_point)
                self.scale_table[node] = scale
        else:
            self.scale_table = [scale] * self.graph.node_size
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        honest_neighbors = self.graph.honest_neighbors[node]
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        mu = torch.mean(local_models[honest_neighbors], dim=0)
        std = torch.std(local_models[honest_neighbors], dim=0)
        melicious_message = mu + self.scale_table[node]*std
        for n in byzantine_neigbors:
            local_models[n].copy_(melicious_message)
        