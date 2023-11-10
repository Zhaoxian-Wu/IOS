import torch
import random
from ByrdLab import DEVICE
from ByrdLab.environment import Dec_Byz_Opt_Env
from ByrdLab.library.measurements import avg_loss_accuracy_dist, consensus_error, one_node_loss_accuracy_dist
from ByrdLab.library.tool import log


class SGD(Dec_Byz_Opt_Env):
    def __init__(self, graph, consensus_init=False, *args, **kw):
        super().__init__(name='SGD', graph=graph, *args, **kw)
        self.consensus_init = consensus_init
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        dist_models = self.construct_dist_models(self.model, self.node_size)
        # self.initilize_models(dist_models, consensus=self.consensus_init)
        # initial record
        loss_path = []
        acc_path = []
        consensus_error_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[SGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, ce={:.5e}, lr={:f}'
        # train_loss = 0
        # train_accuracy = 0
        # total_sample = 0
        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:
                # train_loss_avg = train_loss / total_sample
                # train_accuracy_avg = train_accuracy / total_sample
                test_loss, test_accuracy = avg_loss_accuracy_dist(
                    dist_models, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                ce = consensus_error(dist_models.params_vec,
                                     self.graph.honest_nodes)
                consensus_error_path.append(ce)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, ce, lr
                ))
                # reset the record
                # train_loss_avg = 0
                # train_accuracy_avg = 0
                
            # gradient descent
            for node in self.graph.honest_nodes:
                dist_models.activate_model(node)
                model = dist_models.model
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
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
                
        dist_models.activate_avg_model()
        avg_model = dist_models.model
        return avg_model, loss_path, acc_path, consensus_error_path
    
