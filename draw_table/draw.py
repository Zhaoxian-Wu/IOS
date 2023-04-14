# python draw.py --task SR_mnist --graph TwoCastle_k=6_b=2_seed=40 --partition iidPartition
# python draw.py --task SR_mnist --graph TwoCastle_k=6_b=2_seed=40 --partition LabelSeperation
# python draw.py --task SR_mnist --graph ER_n=12_b=2_p=0.7_seed=300 --partition iidPartition
# python draw.py --task SR_mnist --graph ER_n=12_b=2_p=0.7_seed=300 --partition LabelSeperation

import math
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('..')

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path
from matplotlib import rcParams

import argparse
parser = argparse.ArgumentParser(description='Plotter for robust TD')

parser.add_argument('--task', type=str, default='LR_ijcnn1',
    help='task performed (LR_ijcnn1, SR_mnist, etc)')
parser.add_argument('--graph', type=str, default='ER',
    help='name of the graph (ER, TwoCastle)')
parser.add_argument('--partition', type=str, default='TrivalDist',
    help='name of the partition (TrivalDist)')
parser.add_argument('--workspace', type=str, nargs='+', default=[],
    help='workspace')
parser.add_argument('--mark_on_title', type=str, default='',
    help='mark_on_title')
parser.add_argument('--dont-show-no-comm', action='store_true',
    help='')
parser.add_argument('--table-path', type=str, default='',
    help='path where the table is stored')

args = parser.parse_args()

task_name = args.task
partition_name = args.partition
graph_name = args.graph
workspace = args.workspace
mark_on_title = args.mark_on_title
if mark_on_title != '':
    mark_on_title = '_' + mark_on_title

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

def optimal_gap(path, Fmin):
    return [p-Fmin for p in path]

CLIPPING_UPPER_BOUND = 1e-1
CLIPPING_LOWER_BOUND = 1e-7

pic_name = 'dec_' + task_name + '_' + graph_name + '_' + partition_name

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

if args.partition == 'iidPartition':
    scc_para = ('SCClip_tau=0.1' , 'SCC')
elif args.partition == 'LabelSeperation':
    scc_para = ('SCClip_tau=0.3' , 'SCC')
else:
    assert False
if args.partition == 'iidPartition':
    cc_para = ('CC_tau=0.1' , 'CC')
elif args.partition == 'LabelSeperation':
    cc_para = ('CC_tau=0.3' , 'CC')
else:
    assert False
if args.partition == 'iidPartition':
    rsa_para = ('RSA_lamb=0.001' , 'DRSA')
elif args.partition == 'LabelSeperation':
    rsa_para = ('RSA_lamb=0.5' , 'DRSA')
else:
    assert False
    
aggregations = [
    # ('mean', 'mean'), 
    ('meanW', 'WeiMean'), 
    ('median', 'CooMed'),
    ('geometric_median', 'GeoMed'), 
    ('Krum', 'Krum'), 
    ('trimmed_mean', 'TriMean'),
    ('SimReweight', 'SimRew'),
    rsa_para,
    cc_para,
    scc_para,
    # ('SCClip', 'SCC'),
    # ('SCClip_T', 'SCC-T'),
    ('FABA', 'FABA'), 
    ('IOS', r'\textbf{IOS (ours)}'), 
    # ('bulyan', 'Bulyan'),
    # ('remove_outliers', 'Cutter'),
]

# aggregations = [
#     # ('Centered_Clipping', 'CC'),
#     # ('SCClip', 'SCC'),
#     # ('SCClip_T', 'SCC-T'),
#     # ('SCClip_tau=10', 'SCC'),
#     # ('FABA', 'FABA'), 
#     # ('IOS', r'\textbf{IOS (ours)}'), 
#     # ('IOS_equal_neigbor_weight', 'IOS-equal')
#     # ('RSA_lamb=0.05', 'RSA_lamb=0.05'),
#     # ('RSA_lamb=0.0022', 'RSA_lamb=0.0022'),
#     # ('RSA', 'RSA_lamb=0.001'), # lambda=0.001
    
#     # ('SCClip_tau=100', r'SCC-$\tau=100$'),
#     # ('SCClip_tau=10', r'SCC-$\tau=10$'),
#     # ('SCClip_tau=1', r'SCC-$\tau=1$'),
#     # ('SCClip_tau=0.6', r'SCC-$\tau=0.6$'),
#     # ('SCClip_tau=0.3', r'SCC-$\tau=0.3$'),
#     # ('SCClip_tau=0.1', r'SCC-$\tau=0.1$'),
# ]

attackNames = [
    ('baseline', 'no attack'), 
    ('gaussian', 'Gaussian'), 
    ('sign_flipping', 'sign-flipping'), 
    ('isolation_w', 'isolation'),
    # ('isolation', 'isolation'),
    ('duplicate', 'sample-duplicating'),
    ('alie', 'ALIE'),
]

graph_show_names = {
    'ER_n=12_b=2_p=0.7_seed=300': 'Erdos-Renyi Graph',
    'TwoCastle_k=6_b=2_seed=40': 'Two Castles Graph',
    'Octopus_head=6_headb=0_handb=2': 'Octopus Graph'
}
partition_show_names = {
    'iidPartition': 'i.i.d.',
    'LabelSeperation': 'non-i.i.d.'
}

graph_label_names = {
    'ER_n=12_b=2_p=0.7_seed=300': 'er',
    'TwoCastle_k=6_b=2_seed=40': 'two-castle',
    'Octopus_head=6_headb=0_handb=2': 'octopus'
}
partition_label_names = {
    'iidPartition': 'iid',
    'LabelSeperation': 'non-iid'
}

caption = 'Accuracy (Acc.) and Consensus Error (CE) in ' \
        + graph_show_names[graph_name] + ' in ' \
        + partition_show_names[partition_name] + ' cases.'
        
label = 'table:' + graph_label_names[graph_name] + '-' + partition_label_names[partition_name]

header = r'''\begin{tabular}{'''+ 'c' + '|cc'*len(attackNames) + r'''}
\hline\hline
'''

footer = r'''\hline\hline
\end{tabular}
'''

if args.table_path == '':
    file_dir = os.path.dirname(os.path.abspath(__file__))
    table_path = os.path.join(file_dir, 'table')

    if not os.path.isdir(table_path):
        os.makedirs(table_path)

    table_path = os.path.join(table_path, pic_name + '.tex')
else:
    table_path = args.table_path

acc_table = {}
ce_table = {}

def read_ac_ce(attack_code_name, agg_code_name, mark_on_title,
                task_name, graph_name, partition_name, workspace):
    if agg_code_name[:3] != 'RSA' and agg_code_name[:3] != 'Sim' :
        file_name = 'DSGD_' + attack_code_name + '_' + agg_code_name \
            + '_invSqrtLR' + mark_on_title
    else:
        file_name = agg_code_name + '_' + attack_code_name \
            + '_invSqrtLR' + mark_on_title
    file_path = [task_name, graph_name, partition_name] + workspace
    record = load_file_in_cache(file_name, path_list=file_path)
    loss_path = record['loss_path']
    acc_path = record['acc_path']
    # loss_path = optimal_gap(loss_path, Fmin)
    consensus_error_path = record['consensus_error_path']
    # smooth consensus error path
    smoothed_ce_path = [consensus_error_path[0]]
    smooth_p = 0.1
    for ce in consensus_error_path[1:]:
        smoothed_ce_path.append(smoothed_ce_path[-1]*smooth_p
                                +ce*(1-smooth_p))
        
    acc = sum(acc_path[-10:]) / 10 * 100
    ce = sum(consensus_error_path[-10:]) / 10
    # acc = acc if not math.isnan(acc) and acc < clipping_bound else clipping_bound 
    ce = ce if not math.isnan(ce) and ce < CLIPPING_UPPER_BOUND else CLIPPING_UPPER_BOUND+1
    return acc, ce

for agg_code_name, agg_show_name in aggregations:
    acc_table[agg_code_name] = {}
    ce_table[agg_code_name] = {}
    if agg_code_name == 'no_communication':
        continue
    for attack_code_name, attack_show_name in attackNames:
        acc, ce = read_ac_ce(attack_code_name, agg_code_name, mark_on_title,
                             task_name, graph_name, partition_name, workspace)
        acc_table[agg_code_name][attack_code_name] = acc
        ce_table[agg_code_name][attack_code_name] = ce

acc_color = {
    agg_code_name: {attack_code_name: ('{', '}') for attack_code_name, _ in attackNames}
    for agg_code_name, _ in aggregations
}
ce_color = {
    agg_code_name: {attack_code_name: ('{', '}') for attack_code_name, _ in attackNames}
    for agg_code_name, _ in aggregations
}

# ==============================================================================
# mark the largest and smallest items
for attack_code_name, _ in attackNames:
    acc_order = [
        agg_code_name for agg_code_name, _ in aggregations
    ]
    ce_order = [
        agg_code_name for agg_code_name, _ in aggregations
    ]
    
    acc_order.sort(key=lambda agg_code_name: acc_table[agg_code_name][attack_code_name])
    ce_order.sort(key=lambda agg_code_name: ce_table[agg_code_name][attack_code_name])

    # mark_cnt = 0
    # for i, agg_code_name in enumerate(acc_order):
    #     if i < mark_cnt:
    #         acc_color[agg_code_name][attack_code_name] = (r'\red{', '}')
    #     elif i+mark_cnt >= len(acc_order):
    #         acc_color[agg_code_name][attack_code_name] = (r'\blue{', '}')
    # for i, agg_code_name in enumerate(ce_order):
    #     if i < mark_cnt:
    #         ce_color[agg_code_name][attack_code_name] = (r'\blue{', '}')
    #     elif i+mark_cnt >= len(ce_order):
    #         ce_color[agg_code_name][attack_code_name] = (r'\red{', '}')
    
    # for i, agg_code_name in enumerate(acc_order):
    #     if i+mark_cnt >= len(acc_order):
    #         acc_color[agg_code_name][attack_code_name] = (r'\blue{', '}')
    
    # mark largest
    agg_code_name_largest = acc_order[-1]
    acc_color[agg_code_name_largest][attack_code_name] = (r'\textbf{', '}')
    largest_str = f'{acc_table[agg_code_name_largest][attack_code_name]:.2f}'
    # deal with the situation there are more than 1 larest accuracy
    for agg_code_name in acc_order[-1::-1]:
        acc_str = f'{acc_table[agg_code_name][attack_code_name]:.2f}'
        if largest_str == acc_str:
            acc_color[agg_code_name][attack_code_name] = (r'\textbf{', '}')
        else:
            break
    
# ==============================================================================

with open(table_path, 'w') as f:
    f.write(header)
    # =================================
    header2 = r'\multirow{2}{*}{}'
    for i, (_, attack_show_name) in enumerate(attackNames):
        # header2 += r'&\multicolumn{2}{c' \
        #     + ('|' if i+1<len(attackNames) else '') \
        #     + '}{'+attack_show_name+r'}'
        header2 += r'&\multicolumn{2}{c' \
            + ('' if i+1 == len(attackNames) else '|') \
            + '}{'+attack_show_name+r'}'
    f.write(header2)
    f.write(r'\\')
    f.write('\n')
    
    # =================================
    header3 = r''
    for _, attack_show_name in attackNames:
        header3 += r'& Acc.(\%) & CE '
    f.write(header3)
    f.write(r'\\')
    f.write('\n')
    f.write(r'\hline')
    f.write('\n')

    # =================================
    # no communication
    if not args.dont_show_no_comm:
        line = 'no comm.'
        for attack_code_name, attack_show_name in attackNames:
            if attack_code_name != 'baseline':
                line += ' & -- & -- '
            else:
                acc, ce = read_ac_ce('baseline', 'no_communication', mark_on_title,
                                        task_name, graph_name, partition_name, workspace)
                if ce > CLIPPING_UPPER_BOUND:
                    ce_str = f'$>${CLIPPING_UPPER_BOUND:.0e}'
                elif ce < CLIPPING_LOWER_BOUND:
                    ce_str = f'$<${CLIPPING_LOWER_BOUND:.0e}'
                else:
                    ce_str = f'{ce:.0e}'
                line += ' & ' + f'{acc:.2f}' + ' & ' + ce_str
        f.write(line)
        f.write(r'\\')
        f.write('\n')
        f.write(r'\hline')
        f.write('\n')
        
    # =================================
    # other attack
    for agg_code_name, agg_show_name in aggregations:
        line = agg_show_name
        for attack_code_name, attack_show_name in attackNames:
            if agg_code_name == 'no_communication':
                if attack_code_name != 'baseline':
                    acc, ce  = '--', '--'
                else:
                    acc, ce = read_ac_ce('baseline', 'no_communication', mark_on_title,
                                         task_name, graph_name, partition_name, workspace)        
            else:
                acc = acc_table[agg_code_name][attack_code_name]
                ce = ce_table[agg_code_name][attack_code_name]
            
            acc_color_left, acc_color_right = acc_color[agg_code_name][attack_code_name]
            ce_color_left, ce_color_right = ce_color[agg_code_name][attack_code_name]
            if ce > CLIPPING_UPPER_BOUND:
                ce_str = f'$>${CLIPPING_UPPER_BOUND:.0e}'
            elif ce < CLIPPING_LOWER_BOUND:
                ce_str = f'$<${CLIPPING_LOWER_BOUND:.0e}'
            else:
                ce_str = f'{ce:.0e}'
            line += ' & ' + acc_color_left + f'{acc:.2f}' + acc_color_right \
                + ' & ' + ce_color_left + ce_str + ce_color_right
        f.write(line)
        f.write(r'\\')
        f.write('\n')
            
    f.write(footer)
