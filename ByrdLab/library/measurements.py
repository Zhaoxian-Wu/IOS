import torch

from ByrdLab import TARGET_TYPE, DEVICE

@torch.no_grad()
def consensus_error(local_models, honest_nodes):
    return torch.var(local_models[honest_nodes], 
                     dim=0, unbiased=False).norm().item()

@torch.no_grad()
def avg_loss(model, get_test_iter, loss_fn, weight_decay):
    '''
    The function calculating the loss.
    '''
    loss = 0
    total_sample = 0
    
    # evaluation
    test_iter = get_test_iter()
    for features, targets in test_iter:
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        total_sample += len(targets)
    
    loss_avg = loss / total_sample
    for param in model.parameters():
        loss_avg += weight_decay * param.norm()**2 / 2

    return loss_avg
    
@torch.no_grad()
def avg_loss_accuracy(model, get_test_iter, loss_fn, weight_decay):
    '''
    The function calculating the loss and accuracy.
    '''
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    test_iter = get_test_iter()
    for features, targets in test_iter:
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
def binary_classification_accuracy(predictions, targets):
    prediction_cls = (predictions > 0.5).type(TARGET_TYPE)
    accuracy = (prediction_cls == targets).sum()
    return accuracy

@torch.no_grad()
def multi_classification_accuracy(predictions, targets):
    _, prediction_cls = torch.max(predictions, dim=1)
    accuracy = (prediction_cls == targets).sum()
    return accuracy

@torch.no_grad()
def avg_loss_accuracy_dist(dist_models, get_test_iter,
                           loss_fn, test_fn, weight_decay,
                           node_list=None):
    '''
    The function calculating the loss and accuracy in distributed setting.
    Return the avarage of the local models' accuracy.
    '''
    loss = 0
    accuracy = 0
    total_sample = 0
    
    # evaluation
    # dist_models.activate_avg_model()
    if node_list is None:
        node_list = range(dist_models.node_size)
    for node in node_list:
        dist_models.activate_model(node)
        model = dist_models.model
        test_iter = get_test_iter()
        for features, targets in test_iter:
            # model.to(DEVICE)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            predictions = model(features)
            loss += loss_fn(predictions, targets).item()
            if test_fn is not None:
                accuracy += test_fn(predictions, targets).item()
            total_sample += len(targets)
        
        penalization = 0
        for param in model.parameters():
            penalization += weight_decay * param.norm()**2 / 2
        loss += penalization
    loss_avg = loss / total_sample
    accuracy_avg = accuracy / total_sample

    return loss_avg, accuracy_avg