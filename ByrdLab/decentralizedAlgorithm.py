import torch

from ByrdLab.environment import DecentralizedByzantineEnvironment
from ByrdLab.library import dataset
from ByrdLab.library.dataset import EmptySet
from ByrdLab.DistributedModule import DistributedModule
from ByrdLab.library.partition import EmptyPartition

from .library.measurements import avg_loss_accuracy_dec, consensus_error
from .library.tool import log

# alternative: Apatation & Combination
class DSGD(DecentralizedByzantineEnvironment):
    def __init__(self, graph, aggregation, consensus_init=False, *args, **kw):
        super(DSGD, self).__init__(name='DSGD', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.aggregation = aggregation
    
    def initilize_models(self, dist_models, consensus=False):
        if self.initialize_fn is None:
            return
        for node in self.honest_nodes:
            dist_models.activate_model(node)
            model = dist_models.model
            seed = self.seed if consensus else node + self.seed
            self.initialize_fn(model, fix_init_model=self.fix_seed, seed=seed)
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = DistributedModule(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_dataset[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                # TODO: training loss -> val loss
                val_loss, val_accuracy = avg_loss_accuracy_dec(
                    dist_models, self.get_val_iter, self.loss_fn, 
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(val_loss)
                acc_path.append(val_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    val_loss, val_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # gradient descent
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
                            
            # communication and attack
            self.aggregation.global_state['lr'] = lr
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                # aggregation
                aggregation_res = self.aggregation.run(param_bf_comm, node)
                dist_models.params_vec[node].copy_(aggregation_res)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    
# Adaptation While Combination
class DSGD_AWC(DSGD):
    def __init__(self, graph, aggregation, consensus_init=False, eval_p=-1, *args, **kw):
        super().__init__(graph, consensus_init, eval_p, *args, **kw)
        self.name = 'DSGD_AWC'
        self.aggregation = aggregation
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = DistributedModule(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        grads = [
            [torch.zeros_like(param) for param in dist_models.model.parameters()] 
            for _ in self.graph.nodes
        ]
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_dataset[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                # TODO: training loss -> val loss
                val_loss, val_accuracy = avg_loss_accuracy_dec(
                    dist_models, self.get_val_iter, self.loss_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(val_loss)
                acc_path.append(val_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    val_loss, val_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
                
            # gradient descent
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for (param_idx, param) in enumerate(model.parameters()):
                        if param.grad is not None:
                            param.grad.add_(param, alpha=self.weight_decay)
                            grads[node][param_idx].copy_(param.grad)
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
                            
            # communication and attack
            self.aggregation.global_state['lr'] = lr
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                # aggregation
                aggregation_res = self.aggregation.run(param_bf_comm, node)
                dist_models.params_vec[node].data.copy_(aggregation_res)
                
                with torch.no_grad():
                    dist_models.activate_model(node)
                    model = dist_models.model
                    for (param_idx, param) in enumerate(model.parameters()):
                        param.sub_(grads[node][param_idx], alpha=lr)
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    
      
class RSA_algorithm(DecentralizedByzantineEnvironment):
    def __init__(self, graph, penalty=0.001, consensus_init=False, *args, **kw):
        super(RSA_algorithm, self).__init__(name=f'RSA_lamb={penalty}',
                                            graph=graph, *args, **kw)
        self.consensus_init = consensus_init
        self.lamb = penalty
    
    def initilize_models(self, dist_models, consensus=False):
        if self.initialize_fn is None:
            return
        for node in self.honest_nodes:
            dist_models.activate_model(node)
            model = dist_models.model
            seed = self.seed if consensus else node + self.seed
            self.initialize_fn(model, fix_init_model=self.fix_seed, seed=seed)
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = DistributedModule(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_dataset[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                # TODO: training loss -> val loss
                val_loss, val_accuracy = avg_loss_accuracy_dec(
                    dist_models, self.get_val_iter, self.loss_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(val_loss)
                acc_path.append(val_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    val_loss, val_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
            # main loop
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
                # aggregation
                local_model = dist_models.params_vec[node]
                for j in self.graph.neighbors[node]:
                    diff = param_bf_comm[j] - param_bf_comm[node]
                    local_model.add_(diff.sign(), alpha=lr * self.lamb)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    
    
class Decentralized_gossip(DecentralizedByzantineEnvironment):
    def __init__(self, graph, aggregation, *args, **kw):
        super(Decentralized_gossip, self).__init__(name='gossip',
                                                   model=None,
                                                   dataset=EmptySet(),
                                                   partition_cls=EmptyPartition,
                                                   loss_fn=None,
                                                   lr=0, weight_decay=0, 
                                                   graph=graph, *args, **kw)
        self.aggregation = aggregation
    def run(self, init_local_models = None):
        self.construct_rng_pack()
        local_models = torch.zeros_like(init_local_models)
        local_models.copy_(init_local_models)
        init_ce = consensus_error(local_models, self.graph.honest_nodes)
        ce_path = [init_ce]
        
        for iteration in range(1, self.total_iterations + 1):
            old_local_models = torch.zeros_like(local_models)
            old_local_models.copy_(local_models)
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(old_local_models, node)
                # aggregation
                local_models[node] = self.aggregation.run(old_local_models, node)
            
            if iteration % self.display_interval == 0:
                ce = consensus_error(local_models, self.graph.honest_nodes)
                ce_path.append(ce)
                
        return ce_path
   
    
class DSGD_inner_variation(DSGD):
    def __init__(self, graph, aggregation, inner_variation=0, *args, **kw):
        super().__init__(graph=graph, *args, **kw)
        self.name = f'DSGD_sigma={inner_variation}'
        self.inner_variation = inner_variation
        self.aggregation = aggregation
    
    def initilize_models(self, dist_models, consensus=False):
        if self.initialize_fn is None:
            return
        for node in self.honest_nodes:
            dist_models.activate_model(node)
            model = dist_models.model
            seed = self.seed if consensus else node + self.seed
            self.initialize_fn(model, fix_init_model=self.fix_seed, seed=seed)
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = DistributedModule(self.model, self.node_size)
        self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[DSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # local models before communication
        param_bf_comm = torch.zeros_like(dist_models.params_vec)
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_dataset[node],
                                          rng_pack=self.rng_pack) 
                      if node in self.honest_nodes else None
                      for node in self.nodes]
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                # TODO: training loss -> val loss
                val_loss, val_accuracy = avg_loss_accuracy_dec(
                    dist_models, self.get_val_iter, self.loss_fn, weight_decay=0,
                    node_list=self.honest_nodes)
                
                loss_path.append(val_loss)
                acc_path.append(val_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    val_loss, val_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # gradient descent
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                predictions = model(features)
                loss = self.loss_fn(predictions, targets)
                model.zero_grad()
                loss.backward()
                
                # record loss
                # train_loss += loss.item()
                # train_loss += self.weight_decay / 2 * dist_models.norm(node)**2
                # TODO: correct prediction_cls
                # _, prediction_cls = torch.max(predictions.detach(), dim=1)
                # train_accuracy += (prediction_cls == targets).sum().item()
                # total_sample += len(targets)
                # total_sample += 1
                
                # gradient descend
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.mul_(1 - self.weight_decay * lr)
                            param.data.sub_(param.grad, alpha=lr)
                            noise = torch.randn(param.size(), 
                                                generator=self.rng_pack.torch
                                    ) * self.inner_variation
                            param.data.add_(noise)
            # store the parameters before communication
            param_bf_comm.copy_(dist_models.params_vec)
                            
            # communication and attack
            self.aggregation.global_state['lr'] = lr
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(param_bf_comm, node, self.rng_pack)
                # aggregation
                aggregation_res = self.aggregation.run(param_bf_comm, node)
                dist_models.params_vec[node].copy_(aggregation_res)
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    