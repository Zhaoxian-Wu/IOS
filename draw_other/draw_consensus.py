import matplotlib.pyplot as plt
import torch

from ByrdLab.decentralizedAlgorithm import Decentralized_gossip
from ByrdLab.graph import ErdosRenyi, TwoCastle
from ByrdLab.aggregation import D_centered_clipping, D_mean, D_geometric_median, D_trimmed_mean
from ByrdLab.aggregation import D_Krum, D_median, D_remove_outliers
from ByrdLab.attack import D_sign_flipping
from ByrdLab.library.RandomNumberGenerator import torch_rng

config = {
    'rounds': 20,
    'display_interval': 1,
}

# DRAW_TYPE = 'with_byzantine'
# # DRAW_TYPE = 'no_byzantine'

# if DRAW_TYPE == 'with_byzantine':
#     graph = TwoCastle(3, seed=40)
#     scale = 10
#     rng = torch_generator(20)
#     local_models = scale * torch.rand(graph.node_size, generator=rng).unsqueeze(dim=1)
# elif DRAW_TYPE == 'no_byzantine':
#     graph = TwoCastle(3, byzantine_size=0)
scale = 10
# median fails
graph = ErdosRenyi(10, 1, seed=90)
# graph = ErdosRenyi(8, 0, connected_p=0.7, seed=150)
# graph = TwoCastle(k=3, byzantine_size=0)

rng = torch_rng(20)
local_models = scale * torch.rand(graph.node_size, generator=rng).unsqueeze(dim=1)

attack = D_sign_flipping(graph)

aggregations = [
    D_mean,
    D_median,
    D_geometric_median,
    D_Krum,
    D_trimmed_mean,
]

plt.subplot(1, 2, 1)
graph.show(show_label=True, as_subplot=True, show_lost=True)

plt.subplot(1, 2, 2)
axes = plt.gca()
for aggregation in aggregations:
    aggregation = aggregation(graph)
    optimizer = Decentralized_gossip(aggregation=aggregation, graph=graph,
                                attack=attack, **config)

    ce_path = optimizer.run(init_local_models=local_models)

    x_axis = x_axis = [r*optimizer.display_interval 
                    for r in range(optimizer.rounds+1)]
    axes.plot(x_axis, ce_path, label=f'{aggregation.name}')
SCALE = 6
plt.yscale('log')
plt.legend()
plt.gcf().set_size_inches((SCALE*2.5, SCALE*1))


# if DRAW_TYPE == 'with_byzantine':
#     file_name = './ce-with-byzantine.pdf'
# elif DRAW_TYPE == 'no_byzantine':
#     file_name = './ce-no-byzantine.pdf'

# plt.savefig(file_name, format='pdf', bbox_inches='tight')
plt.show()