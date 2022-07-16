# %%
from ByrdLab.library.cache_io import dump_file_in_cache, isfile_in_cache, load_file_in_cache
import math
import os
import matplotlib.pyplot as plt
import torch
from ByrdLab.library.tool import log
from ByrdLab.aggregation import bulyan, faba, geometric_median, Krum, mKrum, remove_outliers, trimmed_mean, median
from ByrdLab import FEATURE_TYPE
honest_size = 10
X = torch.tensor([
    [math.cos(i*2*math.pi / honest_size), math.sin(i*2*math.pi / honest_size)]
     for i in range(honest_size)
], dtype=FEATURE_TYPE)
x0_h, x1_h = zip(*X)
c_cnt = 60
X_c = torch.tensor([
    [math.cos(i*2*math.pi / c_cnt), math.sin(i*2*math.pi / c_cnt)]
     for i in range(c_cnt)
], dtype=FEATURE_TYPE)
x0_h, x1_h = zip(*X)
x0_c, x1_c = zip(*X_c)
attack_cnt = 100

aggregation_strs = [
    'geometric_median',
    'Krum',
    'mKrum',
    'remove_outliers',
    'FABA',
    'Bulyan',
    'trimmed_mean',
    'median',
]
# p_list = [0.1, 0.3, 0.5, 0.7, 0.9]
p_list = [0.1*i for i in range(1, 10)]

SCALE = 3
# fig.set_size_inches((len(p_list)*SCALE / 2,
#                      len(aggregation_strs)*SCALE))
fig_size = (len(p_list)*SCALE * 0.8, len(aggregation_strs)*SCALE)
fig, big_axes = plt.subplots(figsize=fig_size, 
                             nrows=len(aggregation_strs), ncols=1, sharey=True) 

for row, big_ax in enumerate(big_axes):
    big_ax.set_title(f'{aggregation_strs[row]}\n', fontsize=16)
    # big_ax.set_title('subplot', fontsize=16)

    # Turn off axis lines and ticks of the big subplot 
    # obs alpha is 0 in RGBA string!
    # big_ax.tick_params(labelcolor=(1.,1.,1., 0.0),
    #                    top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax.set_axis_off()
    big_ax._frameon = False
    
# fig, axes_lines = plt.subplots(len(aggregation_strs), len(p_list))
for aggr_index, aggr_str in enumerate(aggregation_strs):
    for p_index, p in enumerate(p_list):
        log(f'{aggr_str} - {p:.1f}')
        byzantine_size = int(honest_size*p)
        node_size = honest_size + byzantine_size

        path_list = ['showAggregation']
        file_name = aggr_str + f'_byz={p:.1f}'
        if isfile_in_cache(file_name, path_list=path_list):
            X_aggr = load_file_in_cache(file_name, path_list=path_list)
        else:
            X_aggr = []
            for _ in range(attack_cnt):
                att = 20*torch.randn(byzantine_size, 2)
                local_models = torch.cat([X, att])
                if aggr_str == 'geometric_median':
                    agg = geometric_median(local_models)
                elif aggr_str == 'Krum':
                    agg = Krum(local_models, byzantine_size)
                elif aggr_str == 'mKrum':
                    agg = mKrum(local_models, byzantine_size, 
                                m=max(node_size-2*byzantine_size-2, 1))
                elif aggr_str == 'remove_outliers':
                    agg = remove_outliers(local_models, byzantine_size)
                elif aggr_str == 'FABA':
                    agg = faba(local_models, byzantine_size)
                elif aggr_str == 'trimmed_mean':
                    agg = trimmed_mean(local_models, byzantine_size)
                elif aggr_str == 'median':
                    agg = median(local_models)
                elif aggr_str == 'Bulyan':
                    if node_size > 4*byzantine_size+3:
                        agg = bulyan(local_models, byzantine_size)
                    else:
                        agg = local_models.mean(dim=0)
                X_aggr.append(agg)
            dump_file_in_cache(file_name, X_aggr, path_list=path_list)
        x0, x1 = zip(*X_aggr)
        ax = fig.add_subplot(len(aggregation_strs), len(p_list), 
                             aggr_index*len(p_list) + p_index + 1)
        # ax = axes_lines[aggr_index][p_index]
        ax.plot(x0_c, x1_c, '--')
        ax.scatter(x0_h, x1_h)
        ax.scatter(x0, x1)
        ax.set_axis_off()
        ax.set_title(f'byz={p:.1f}')
fig.tight_layout()

pic_path = os.path.join('pic', 'showAggregation.pdf')
plt.savefig(pic_path, format='pdf', bbox_inches='tight')
plt.show()
