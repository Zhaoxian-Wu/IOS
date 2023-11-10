from argsParser import args
from scipy.optimize import linprog
from functools import partial
import numpy as np
from gurobipy import *

from ByrdLab import DEVICE
from ByrdLab.SingleMachineAlgorithm import SGD
from ByrdLab.graph import CompleteGraph, ErdosRenyi, OctopusGraph, TwoCastle
from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab.library.cache_io import dump_file_in_cache, dump_model_in_cache, load_model_in_cache
from ByrdLab.library.dataset import ijcnn, mnist, fashionmnist, cifar10
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition)
from ByrdLab.library.tool import log
from ByrdLab.tasks.logisticRegression import LogisticRegressionTask
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask, random_generator, softmax_regression_loss


# run for decentralized algorithm
# -------------------------------------------
# define graph
# -------------------------------------------
graph = CompleteGraph(node_size=1, byzantine_size=0)

# ===========================================

# -------------------------------------------
# define learning task
# -------------------------------------------
# data_package = ijcnn()
# task = LogisticRegressionTask(data_package)

# dataset = ToySet(set_size=500, dimension=5, fix_seed=True)

data_package = mnist()
task = softmaxRegressionTask(data_package)

# data_package = fashionmnist()
# task = softmaxRegressionTask(data_package)

# data_package = cifar10()
# task = NeuralNetworkTask(data_package, batch_size=32)

# ===========================================


# -------------------------------------------
# define learning rate control rule
# -------------------------------------------
if args.lr_ctrl == 'constant':
    lr_ctrl = None
elif args.lr_ctrl == '1/sqrt k':
    lr_ctrl = one_over_sqrt_k_lr(a=1, b=1)
    # super_params = task.super_params
    # total_iterations = super_params['rounds']*super_params['display_interval']
    # lr_ctrl = one_over_sqrt_k_lr(total_iteration=total_iterations,
    #                              a=math.sqrt(1001), b=1000)
elif args.lr_ctrl == 'ladder':
    decreasing_iter_ls = [5000, 10000, 15000]
    proportion_ls = [0.3, 0.2, 0.1]
    lr_ctrl = ladder_lr(decreasing_iter_ls, proportion_ls)
else:
    assert False, 'unknown lr-ctrl'

# ===========================================
    
    
# -------------------------------------------
# define data partition
# -------------------------------------------
partition_cls = iidPartition


# ===========================================
    

# -------------------------------------------
# define aggregation
# -------------------------------------------

# ===========================================
    
# -------------------------------------------
# define attack
# -------------------------------------------
attack = None
attack_name = 'baseline'


# ===========================================

# workspace = []
workspace = ['best']
mark_on_title = ''
fix_seed = not args.no_fixed_seed
seed = args.seed
record_in_file = not args.without_record
step_agg = args.step_agg

# ===========================================
def solveLP(loss_clean_model, loss_poisoned_model):
    C = data_size // 10
    func_coeff = np.zeros(len_U)
    for i in range(len_U):
        func_coeff[i] = loss_poisoned_model[i] - loss_clean_model[i]

    # constraints
    A_ub = [0] * data_size + [1] * (len_U - data_size)
    b_ub = [C]
    A_eq = []
    b_eq = []

    for i in range(data_size):
        temp = [0] * len_U
        for k in range(num_classes):
            temp[i + k * data_size] = 1
        A_eq.append(temp)
        b_eq.append(1)
    
    Q_bound = tuple([(0, 1)] * len_U)

    q = linprog(func_coeff, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = Q_bound, options={"disp": False, "maxiter": 10000}).x
    return q


def solveLP_by_gurobi(loss_clean_model, loss_poisoned_model):
    C = data_size // 10
    try:
        model=Model('mip')
        q = model.addMVar((len_U), lb=0, ub=1, vtype=GRB.CONTINUOUS, name='q')

        func_coeff = loss_poisoned_model - loss_clean_model

        # constraints
        A_ub = [0] * data_size + [1] * (len_U - data_size)
        b_ub = [C]
        # A_eq = []
        # b_eq = []

        # for i in range(data_size):
        #     temp = [0] * len_U
        #     for k in range(num_classes):
        #         temp[i + k * data_size] = 1
        #     A_eq.append(temp)
        #     b_eq.append(1)

        A_ub = np.array(A_ub)
        # A_eq = np.array(A_eq)
        # b_eq = np.array(b_eq)

        # import scipy.sparse
        # A_eq = scipy.sparse.csr_matrix(A_eq)

        # 目标函数
        model.setObjective(func_coeff @ q, GRB.MINIMIZE)

        # 约束
        model.addConstr(A_ub @ q <= C, name='c1')
        # model.addConstr(A_eq @ q == b_eq, name='c2')
        # for i in range(data_size):
        #     A_eq = np.zeros(len_U)
        #     for k in range(num_classes):
        #         A_eq[i + k * data_size] = 1
        #     model.addConstr(A_eq @ q == 1)
        
        chunk_size = 10000  # Define a chunk size to add constraints in smaller batches

        for start_index in range(0, data_size, chunk_size):
            end_index = min(start_index + chunk_size, data_size)

            constraint_matrix = np.zeros((end_index - start_index, num_classes * data_size))
            for i in range(start_index, end_index):
                for k in range(num_classes):
                    constraint_matrix[i - start_index, i + k * data_size] = 1

            model.addConstrs((constraint_matrix[j, :] @ q == 1 for j in range(end_index - start_index)))

        
        log('[Start Solving LP]')
        #求解
        model.setParam('outPutFlag', 1)#不输出求解日志
        model.setParam('Method', 1)  # Switch to primal simplex
        model.setParam('PreCrush', 1)  # Enable presolve
        # model.setParam('WarmStart', 1)  # Enable warm start
        # model.setParam('Threads', 24)  # Set the number of threads
        model.optimize()

        #输出
        print('obj=',model.objVal)
        for v in model.getVars():
            if v.x != 0:
                print(v.varName,':',v.x)
    except GurobiError as e:
        print('Error code '+str(e.errno)+':'+str(e))

    except AttributeError:
        print('Encountered an attribute error')
    return q


# ===========================================
path_best_list = [task.name, graph.name, "iidPartition"] + ["best"]
best_title = "SGD_baseline_invSqrtLR.pt"
best_clean_model = load_model_in_cache(best_title,  path_list=path_best_list)

get_train_iter = partial(random_generator, batch_size=1)
data_size = len(data_package.train_set)
num_classes = data_package.num_classes
len_U = 10  * data_size

loss_fn = softmax_regression_loss
loss_clean_model = np.zeros(len_U)
loss_poisoned_model = np.zeros(len_U)
train_iter = get_train_iter(dataset=data_package.train_set, rng_pack=RngPackage(seed=seed))
for i in range(data_size):
    feature, target = next(train_iter)
    feature = feature.to(DEVICE)
    target = target.to(DEVICE)
    prediction = best_clean_model(feature)
    for k in range(num_classes):
        target = (target + k) % num_classes
        loss_clean_model[i + k * data_size] = loss_fn(prediction, target)
        # print(loss_clean_model[i + k * data_size])

path_list = [task.name, graph.name, "iidPartition"] + workspace
dump_file_in_cache("loss_best_model_on_each_sample", loss_clean_model, path_list=path_list)

# ===========================================
# initilize optimizer
env = SGD(graph=graph, attack=attack, step_agg = step_agg,
           weight_decay=task.weight_decay, data_package=task.data_package,
           model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
           initialize_fn=task.initialize_fn,
           get_train_iter=task.get_train_iter,
           get_test_iter=task.get_test_iter,
           partition_cls=partition_cls, lr_ctrl=lr_ctrl,
           fix_seed=fix_seed, seed=seed,
           **task.super_params)


title = '{}_{}'.format(env.name, attack_name)

if lr_ctrl != None:
    title = title + '_' + lr_ctrl.name
if mark_on_title != '':
    title = title + '_' + mark_on_title

data_package = task.data_package
super_params = task.super_params

# print the running information
print('=========================================================')
print('[Task] ' + task.name + ': ' + title)
print('=========================================================')
print('[Setting]')
print('{:12s} model={}'.format('[task]', task.model_name))
print('{:12s} dataset={} partition={}'.format(
    '[dataset]', data_package.name, env.partition_name))
print('{:12s} name={} aggregation={} attack={}'.format(
    '[Algorithm]', env.name, 'None', attack_name))
print('{:12s} lr={} lr_ctrl={}, weight_decay={}'.format(
    '[Optimizer]', super_params['lr'], env.lr_ctrl.name, task.weight_decay))
print('{:12s} graph={}, honest_size={}, byzantine_size={}'.format(
    '[Graph]', graph.name, graph.honest_size, graph.byzantine_size))
print('{:12s} rounds={}, display_interval={}, total iterations={}'.format(
    '[Running]', env.rounds, env.display_interval, env.total_iterations))
print('{:12s} seed={}, fix_seed={}'.format('[Randomness]', seed, fix_seed))
print('{:12s} record_in_file={}'.format('[System]', record_in_file))
print('-------------------------------------------')

log('[Start Running]')
q = solveLP_by_gurobi(loss_clean_model=loss_clean_model, loss_poisoned_model=loss_poisoned_model)
dump_file_in_cache("q-start", q, path_list=path_list)
# maxIter = 10
# for _ in range(maxIter):
#     q = solveLP(loss_clean_model=loss_clean_model, loss_poisoned_model=loss_poisoned_model)
    # avg_model, loss_path, acc_path, consensus_error_path = env.run()


# record = {
#     'dataset': data_package.name,
#     'dataset_size': len(data_package.train_set),
#     'dataset_feature_dimension': data_package.feature_dimension,
#     'lr': super_params['lr'],
#     'weight_decay': task.weight_decay,
#     'honest_size': graph.honest_size,
#     'byzantine_size': graph.byzantine_size,
#     'rounds': env.rounds,
#     'display_interval': env.display_interval,
#     'total_iterations': env.total_iterations,
#     'loss_path': loss_path,
#     'acc_path': acc_path,
#     'consensus_error_path': consensus_error_path,
#     'fix_seed': fix_seed,
#     'seed': seed,
#     'graph': graph,
# }

# if record_in_file:
#     path_list = [task.name, graph.name, env.partition_name] + workspace
#     dump_file_in_cache(title, record, path_list=path_list)
log('-------------------------------------------')

