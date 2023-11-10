import argparse

from ByrdLab.attack import (D_alie, D_gaussian, D_isolation_weight,
                            D_sample_duplicate, D_sign_flipping,
                            D_zero_sum, D_zero_value, D_label_flipping, D_label_random, D_feature_label_random)
from ByrdLab.decentralizedAlgorithm import RSA_algorithm, RSA_algorithm_under_DPA
from ByrdLab.graph import CompleteGraph, ErdosRenyi, OctopusGraph, TwoCastle, LineGraph
from ByrdLab.library.cache_io import dump_file_in_cache
from ByrdLab.library.dataset import mnist
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition)
from ByrdLab.library.tool import log
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.tasks.leastSquare import LeastSquareToySet, LeastSquareToyTask

parser = argparse.ArgumentParser(description='Robust Temporal Difference Learning')
    
# Arguments
parser.add_argument('--graph', type=str, default='CompleteGraph')
parser.add_argument('--attack', type=str, default='none')
parser.add_argument('--data-partition', type=str, default='iid')
parser.add_argument('--lr-ctrl', type=str, default='1/sqrt k')

parser.add_argument('--no-fixed-seed', action='store_true',
                    help="If specifed, the random seed won't be fixed")
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--without-record', action='store_true',
                    help='If specifed, no file of running record and log will be left')

args = parser.parse_args()

args.attack = 'label_flipping'
args.graph = 'CompleteGraph'
args.data_partition = 'noniid'

# run for decentralized algorithm
# -------------------------------------------
# define graph
# -------------------------------------------
if args.graph == 'CompleteGraph':
    graph = CompleteGraph(node_size=10, byzantine_size=2)
elif args.graph == 'TwoCastle':
    graph = TwoCastle(k=6, byzantine_size=2, seed=40)
elif args.graph == 'ER':
    honest_size = 100
    byzantine_size = 2
    node_size = honest_size + byzantine_size
    graph = ErdosRenyi(node_size, byzantine_size, seed=300)
elif args.graph == 'LineGraph':
    honest_size = 2000
    byzantine_size = 1
    node_size = honest_size + byzantine_size
    graph = LineGraph(node_size=node_size, byzantine_size=byzantine_size)
elif args.graph == 'OctopusGraph':
    graph = OctopusGraph(6, 0, 2)
else:
    assert False, 'unknown graph'
    
if args.attack == 'none':
    graph = graph.honest_subgraph()
# ===========================================

# -------------------------------------------
# define learning task
# -------------------------------------------
# dataset = ijcnn()
# dataset = ToySet(set_size=500, dimension=5, fix_seed=True)

data_package = mnist()
task = softmaxRegressionTask(data_package)

# data_package = LeastSquareToySet(set_size=2000, dimension=1, noise=0, fix_seed=True)
# task = LeastSquareToyTask(data_package)

# task.super_params['display_interval'] = 20000
# task.super_params['rounds'] = 10
# task.super_params['lr'] = 0.004
# task.super_params['lr'] = 0.1
# task.initialize_fn = None
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
    decreasing_iter_ls = [30000, 60000]
    proportion_ls = [0.5, 0.2]
    lr_ctrl = ladder_lr(decreasing_iter_ls, proportion_ls)
else:
    assert False, 'unknown lr-ctrl'
# ===========================================
    
    
# -------------------------------------------
# define data partition
# -------------------------------------------
if args.data_partition == 'trival':
    partition_cls = TrivalPartition
elif args.data_partition == 'iid':
    partition_cls = iidPartition
elif args.data_partition == 'noniid':
    partition_cls = LabelSeperation
else:
    assert False, 'unknown data-partition'
# ===========================================
    
    
# -------------------------------------------
# define atack
# -------------------------------------------
if args.attack == 'none':
    attack = None
elif args.attack == 'label_flipping':
    attack = D_label_flipping(graph)
elif args.attack == 'label_random':
    attack = D_label_random(graph)
elif args.attack == 'feature_label_random':
    attack = D_feature_label_random(graph)
elif args.attack == 'sign_flipping':
    attack = D_sign_flipping(graph)
elif args.attack == 'gaussian':
    attack = D_gaussian(graph)
elif args.attack == 'isolation':
    # D_isolation(graph),
    attack = D_isolation_weight(graph)
elif args.attack == 'sample_duplicate':
    attack = D_sample_duplicate(graph)
elif args.attack == 'zero_sum':
    attack = D_zero_sum(graph)
elif args.attack == 'zero_value':
    attack = D_zero_value(graph)
elif args.attack == 'alie':
    attack = D_alie(graph)
    # D_alie(graph, scale=2)
else:
    assert False, 'unknown graph'

if args.attack == 'none':
    attack_name = 'baseline'
else:
    attack_name = attack.name
# ===========================================


workspace = []
mark_on_title = ''
fix_seed = not args.no_fixed_seed
seed = args.seed
record_in_file = not args.without_record


if args.data_partition == 'iid':
    penalty = 0.001
elif args.data_partition == 'noniid':
    penalty = 0.1
else:
    penalty = 0.5
    
# initilize optimizer
if 'label' in attack_name:
    env = RSA_algorithm_under_DPA(graph=graph, attack=attack,
                                  weight_decay=task.weight_decay,
                                  data_package=task.data_package,
                                  model=task.model,
                                  loss_fn=task.loss_fn, test_fn=task.test_fn,
                                  initialize_fn=task.initialize_fn,
                                  get_train_iter=task.get_train_iter,
                                  get_test_iter=task.get_test_iter,
                                  partition_cls=partition_cls, lr_ctrl=lr_ctrl,
                                  fix_seed=fix_seed, seed=seed,
                                  penalty=penalty,
                                  **task.super_params)
else:
    env = RSA_algorithm(graph=graph, attack=attack,
                        weight_decay=task.weight_decay,
                        data_package=task.data_package,
                        model=task.model,
                        loss_fn=task.loss_fn, test_fn=task.test_fn,
                        initialize_fn=task.initialize_fn,
                        get_train_iter=task.get_train_iter,
                        get_test_iter=task.get_test_iter,
                        partition_cls=partition_cls, lr_ctrl=lr_ctrl,
                        fix_seed=fix_seed, seed=seed,
                        penalty=penalty,
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
print('{:12s} name={} attack={}'.format(
    '[Algorithm]', env.name, attack_name))
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
_, loss_path, acc_path, consensus_error_path = env.run()

record = {
    'dataset': data_package.name,
    'dataset_size': len(data_package.train_set),
    'dataset_feature_dimension': data_package.feature_dimension,
    'lr': super_params['lr'],
    'weight_decay': task.weight_decay,
    'honest_size': graph.honest_size,
    'byzantine_size': graph.byzantine_size,
    'rounds': env.rounds,
    'display_interval': env.display_interval,
    'total_iterations': env.total_iterations,
    'loss_path': loss_path,
    'acc_path': acc_path,
    'consensus_error_path': consensus_error_path,
    'fix_seed': fix_seed,
    'seed': seed,
    'graph': graph,
}

if record_in_file:
    path_list = [task.name, graph.name, env.partition_name] + workspace
    dump_file_in_cache(title, record, path_list=path_list)
print('-------------------------------------------')


