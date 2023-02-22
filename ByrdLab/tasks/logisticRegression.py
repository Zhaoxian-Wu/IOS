from functools import partial
from ByrdLab.library.dataset import DataPackage
from ByrdLab.library.initialize import RandomInitialize
from ByrdLab.library.measurements import binary_classification_accuracy
from ByrdLab.tasks import Task
import torch
import torch.autograd

from ByrdLab import FEATURE_TYPE, TARGET_TYPE
from ByrdLab.tasks.softmaxRegression import order_generator, random_generator

num_classes = 2

class LogisticRegressionTask(Task):
    def __init__(self, data_package: DataPackage, batch_size=32):
        weight_decay = 0.01
        model = logisticRegression_model(data_package.feature_dimension)
        loss_fn = logistic_regression_loss
        test_fn = binary_classification_accuracy
        super_params = {
            'rounds': 10,
            'display_interval': 4000,
            'batch_size': batch_size,
            'test_batch_size': 900,
            
            'lr': 2e-2,
        }
        
        test_set = data_package.test_set
        get_train_iter = partial(random_generator,
                                 batch_size=super_params['batch_size'])
        get_test_iter = partial(order_generator, dataset=test_set,
                                 batch_size=super_params['test_batch_size'])
        super().__init__(weight_decay, data_package, model, loss_fn, test_fn,
                         initialize_fn=RandomInitialize(),
                         get_train_iter=get_train_iter,
                         get_test_iter=get_test_iter,
                         super_params=super_params,
                         name=f'LR_{data_package.name}',
                         model_name='LogisticRegression')

class logisticRegression_model(torch.nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=feature_dimension,
                                      out_features=1, bias=True)
    def forward(self, features):
        z = self.linear(features)
        p = torch.sigmoid(z).view(-1)
        return p

def logistic_regression_loss(predictions, targets):
    loss = torch.nn.functional.binary_cross_entropy(
                predictions, targets.type_as(predictions).view(-1))
    return loss
    
def backward_closure(predictions, targets):
    def _backward():
        # feature = self.feature
        # predictions = self.predictions
        # targets = self.targets
        feature = predictions.feature
        model = predictions.model
        p = predictions[0][0]
        
        err = (p - targets).data
        params = list(model.parameters())
        params[0].grad = err * feature.view((1, -1))
        params[1].grad = err * torch.ones(1, dtype=FEATURE_TYPE)
    return _backward
    
def logistic_regression(model, x):
    out = model[:-1].dot(x) + model[-1]
    return torch.sigmoid(out)

def accuracy(model, dataset):
    correct = 0
    for feature, target in dataset:
        predict = logistic_regression(model, feature) > 0.5
        correct += (predict.type(TARGET_TYPE) == target).item()
    return correct / len(dataset)

def gradient_sto(model, data, weight_decay):  
    feature, target = data
    predict = logistic_regression(model, feature)

    err = -(target-predict).data
    g = err*torch.cat([feature, torch.ones(1)])
    g.add_(model, alpha=weight_decay)
    return g
    
def gradient_avg(model, dataset, weight_decay):
    '''
    return averaged gradient over the whole dataset
    '''
    G = torch.zeros_like(model, requires_grad=False, dtype=FEATURE_TYPE)
    for feature, target in dataset:
        predict = logistic_regression(model, feature)

        err = -(target-predict).data
        G[:-1].add_(feature, alpha=err/len(dataset))
        G[-1].add_(1, alpha=err/len(dataset))
    G.add_(model, alpha=weight_decay)
    return G

def get_varience(w_local, honestSize):
    avg = w_local[:honestSize].mean(dim=0)
    s = 0
    for w in w_local[:honestSize]:
        s += (w - avg).norm()**2
    s /= honestSize
    return s.item()

def get_outer_variation(w_min, dataset, honestSize):
    # data split
    pieces = [(i*len(dataset)) // honestSize for i in range(honestSize+1)]
    data_per_node = [pieces[i+1] - pieces[i] for i in range(honestSize)]
    gradients = []
    for node in range(honestSize):
        gradient = torch.zeros_like(w_min)
        for index in range(pieces[node], pieces[node+1]):
            x, y = dataset[index]
            predict = logistic_regression(w_min, x)

            err = (predict-y).data
            gradient[:-1].add_(err*x)
            gradient[-1].add_(err)
        gradient.div_(data_per_node[node])
        gradients.append(gradient)
    gradients = torch.stack(gradients)
    outerVariation = get_varience(gradients, honestSize)
    return outerVariation