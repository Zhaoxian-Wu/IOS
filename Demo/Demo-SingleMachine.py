from ByrdLab.library.RandomNumberGenerator import random_rng

import torch
from ByrdLab.library.learnRateController import one_over_sqrt_k_lr
from ByrdLab.library.tool import log
from torch.utils import data
from ByrdLab.library.measurements import avg_loss_accuracy
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.library.dataset import mnist

dataset = mnist()
task = softmaxRegressionTask(dataset)
task.super_params['display_interval'] = 100
task.super_params['rounds'] = 20

lr_ctrl = one_over_sqrt_k_lr(a=1, b=1)
# lr_ctrl = constant_lr(task.super_params['lr'])
lr_ctrl.set_init_lr(task.super_params['lr'])

# initialize
model = task.model
loss_fn = task.loss_fn
weight_decay = task.weight_decay
display_interval = task.super_params['display_interval']
rounds = task.super_params['rounds']
total_iterations = display_interval * rounds
get_train_iter = task.get_train_iter
get_val_iter = task.get_val_iter
rng = random_rng(100)

# log formatter
num_len = len(str(total_iterations))
num_format = '{:>' + f'{num_len}' + 'd}'
hint = '[SGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
    'loss={:.3e}, accuracy={:.2f}, lr={:f}'
# train_loss = 0
# train_accuracy = 0
# total_sample = 0

data_iter = get_train_iter(dataset=dataset, rng=rng)
for iteration in range(0, total_iterations + 1):
    # lastest learning rate
    lr = lr_ctrl.get_lr(iteration)
    
    # record (totally 'rounds+1' times)
    if iteration % display_interval == 0:
        # train_loss_avg = train_loss / total_sample
        # train_accuracy_avg = train_accuracy / total_sample
        # TODO: training loss -> val loss
        val_iter = get_val_iter()
        train_loss, train_accuracy = avg_loss_accuracy(
            model, val_iter, loss_fn, weight_decay=0)
        
        log(hint.format(
            iteration, total_iterations,
            iteration / total_iterations * 100,
            train_loss, train_accuracy, lr
        ))
        # reset the record
        # train_loss_avg = 0
        # train_accuracy_avg = 0
        
    # gradient descent
    features, targets = next(data_iter)
    predictions = model(features)
    loss = loss_fn(predictions, targets)
    model.zero_grad()
    loss.backward()
    
    # record loss
    # train_loss += loss.item()
    # train_loss += weight_decay / 2 * dist_models.norm(node)**2
    # TODO: correct prediction_cls
    # _, prediction_cls = torch.max(predictions.detach(), dim=1)
    # train_accuracy += (prediction_cls == targets).sum().item()
    # total_sample += len(targets)
    # total_sample += 1
    
    # gradient descend
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data.mul_(1 - weight_decay * lr)
                param.data.sub_(param.grad, alpha=lr)
                