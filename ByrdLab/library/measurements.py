from random import random
import torch
from torch.utils import data

def consensus_error(local_models, honest_nodes):
    return torch.var(local_models[honest_nodes], 
                     dim=0, unbiased=False).norm().item()
    
@torch.no_grad()
def avg_loss_accuracy(model, val_iter, loss_fn, weight_decay):
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    for features, targets in val_iter:
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        _, prediction_cls = torch.max(predictions.detach(), dim=1)
        accuracy += (prediction_cls == targets).sum().item()
        total_sample += len(targets)
    
    loss_avg = loss / total_sample
    accuracy_avg = accuracy / total_sample
    for param in model.parameters():
        loss_avg += weight_decay * param.norm()**2 / 2

    return loss_avg, accuracy_avg

@torch.no_grad()
def avg_loss_accuracy_dec(dist_models, val_iter, loss_fn, weight_decay,
                          node_list=None):
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    # dist_models.activate_avg_model()
    # TODO: debug
    if node_list is None:
        node_list = range(dist_models.node_size)
    for node in node_list:
        dist_models.activate_model(node)
        model = dist_models.model
        for features, targets in val_iter:
            predictions = model(features)
            loss += loss_fn(predictions, targets).item()
            _, prediction_cls = torch.max(predictions.detach(), dim=1)
            accuracy += (prediction_cls == targets).sum().item()
            total_sample += len(targets)
        
    penalization = 0
    for param in model.parameters():
        penalization += weight_decay * param.norm()**2 / 2
    loss_avg = loss / total_sample + penalization
    accuracy_avg = accuracy / total_sample

    return loss_avg, accuracy_avg

@torch.no_grad()
def avg_residual(model, val_iter, loss_fn, weight_decay):
    residual = 0
    total_sample = 0
    
    # evaluation
    for features, targets in val_iter:
        predictions = model(features)
        residual = loss_fn(predictions, targets)
        total_sample += len(targets)
    
    residual_avg = residual / total_sample
    for param in model.parameters():
        residual_avg += weight_decay * param.norm()**2 / 2

    return residual_avg
    