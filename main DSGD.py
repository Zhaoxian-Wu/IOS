import argparse

from ByrdLab.aggregation import (D_bulyan, D_faba, D_geometric_median, D_Krum, D_ios_equal_neigbor_weight, D_mean,
                                 D_meanW, D_median, D_no_communication,
                                 D_remove_outliers, D_self_centered_clipping, D_trimmed_mean,
                                 D_mKrum, D_centered_clipping,
                                 D_ios)
from ByrdLab.attack import (D_alie, D_gaussian, D_isolation, D_isolation_weight,
                            D_sample_duplicate, D_sign_flipping,
                            D_zero_sum, D_zero_value)
from ByrdLab.decentralizedAlgorithm import DSGD
from ByrdLab.graph import CompleteGraph, ErdosRenyi, OctopusGraph, TwoCastle
from ByrdLab.library.cache_io import dump_file_in_cache
from ByrdLab.library.dataset import mnist
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition)
from ByrdLab.library.tool import log, no_exception_blocking
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask

parser = argparse.ArgumentParser(description='Robust Temporal Difference Learning')
    
# Arguments
parser.add_argument('--graph', type=str, default='CompleteGraph')
parser.add_argument('--aggregation', type=str, default='mean')
parser.add_argument('--attack', type=str, default='none')
parser.add_argument('--data-partition', type=str, default='iid')
parser.add_argument('--lr-ctrl', type=str, default='1/sqrt k')

parser.add_argument('--fix-seed', type=bool, default=True)
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--record-in-file', type=bool, default=True)

args = parser.parse_args()


# run for decentralized algorithm
# -------------------------------------------
# define graph
# -------------------------------------------
if args.graph == 'CompleteGraph':
    graph = CompleteGraph(node_size=12, byzantine_size=2)
elif args.graph == 'TwoCastle':
    graph = TwoCastle(k=6, byzantine_size=2, seed=40)
elif args.graph == 'ER':
    honest_size = 10
    byzantine_size = 2
    node_size = honest_size + byzantine_size
    graph = ErdosRenyi(node_size, byzantine_size, seed=300)
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
dataset = mnist()
task = softmaxRegressionTask(dataset)
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
# define aggregation
# -------------------------------------------
if args.aggregation == 'no-comm':
    aggregation = D_no_communication(graph)
elif args.aggregation == 'mean':
    # D_mean(graph),
    aggregation = D_meanW(graph)
elif args.aggregation == 'ios':
    aggregation = D_ios(graph)
elif args.aggregation == 'ios_exact_byz_cnt':
    aggregation = D_ios(graph, exact_byz_cnt=False, byz_cnt=2)
elif args.aggregation == 'ios_equal_neigbor_weight':
    aggregation = D_ios_equal_neigbor_weight(graph)
elif args.aggregation == 'trimmed-mean':
    aggregation = D_trimmed_mean(graph)
elif args.aggregation == 'median':
    aggregation = D_median(graph)
elif args.aggregation == 'geometric-median':
    aggregation = D_geometric_median(graph)
elif args.aggregation == 'faba':
    aggregation = D_faba(graph)
elif args.aggregation == 'remove-outliers':
    aggregation = D_remove_outliers(graph)
elif args.aggregation == 'mKrum':
    aggregation = D_mKrum(graph)
elif args.aggregation == 'Krum':
    aggregation = D_Krum(graph)
elif args.aggregation == 'bulyan':
    aggregation = D_bulyan(graph)
elif args.aggregation == 'cc':
    if args.data_partition == 'iid':
        threshold = 0.1
    elif args.data_partition == 'noniid':
        threshold = 0.3
    else:
        threshold = 0.3
    aggregation = D_centered_clipping(graph, threshold=threshold)
    # D_self_centered_clipping(graph, threshold_selection='true'),
    # D_self_centered_clipping(graph),
elif args.aggregation == 'scc':
    if args.data_partition == 'iid':
        threshold = 0.1
    elif args.data_partition == 'noniid':
        threshold = 0.3
    else:
        threshold = 0.3
    aggregation = D_self_centered_clipping(
        graph, threshold_selection='parameter', threshold=threshold)
    # D_self_centered_clipping(graph, threshold_selection='true'),
else:
    assert False, 'unknown aggregation'
# ===========================================
    
# -------------------------------------------
# define attack
# -------------------------------------------
if args.attack == 'none':
    attack = None
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

if args.attack == 'none':
    attack_name = 'baseline'
else:
    attack_name = attack.name
# ===========================================

workspace = []
mark_on_title = ''
# workspace=['rsa_tuning']
# mark_on_title='test'
fix_seed = args.fix_seed
seed = args.seed
record_in_file = args.record_in_file

# initilize optimizer
env = DSGD(aggregation=aggregation, graph=graph, attack=attack,
            weight_decay=task.weight_decay, dataset=task.dataset,
            model=task.model, loss_fn=task.loss_fn,
            initialize_fn=task.initialize_fn,
            get_train_iter=task.get_train_iter,
            get_val_iter=task.get_val_iter,
            partition_cls=partition_cls, lr_ctrl=lr_ctrl,
            fix_seed=fix_seed, seed=seed,
            #   eval_p=1,
            **task.super_params)

title = '{}_{}_{}'.format(env.name, attack_name, aggregation.name)

if lr_ctrl != None:
    title = title + '_' + lr_ctrl.name
if mark_on_title != '':
    title = title + '_' + mark_on_title

dataset = task.dataset

super_params = task.super_params
# print the running information
print('=========================================================')
print('[Task] ' + task.name + ': ' + title)
print('=========================================================')
print('[Setting]')
print('{:12s} model={}'.format('[task]', task.model_name))
print('{:12s} dataset={} partition={}'.format(
    '[dataset]', dataset.name, env.partition_name))
print('{:12s} name={} aggregation={} attack={}'.format(
    '[Algorithm]', env.name, aggregation.name, attack_name))
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
    'dataset': dataset.name,
    'dataset_size': len(dataset),
    'dataset_feature_dimension': dataset.feature_dimension,
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

