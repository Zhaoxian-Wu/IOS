# %%
import math
import os
import matplotlib.pyplot as plt
import torch
from ByrdLab import FEATURE_TYPE
honest_size = 3
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

repeat_cnt = 1000
combination_list = []
w_bound = 0.6
# for _ in range(repeat_cnt):
#     w = torch.rand(honest_size)
#     w = w / w.sum()
#     if w.max() < w_bound:
#         combination_list.append(torch.tensordot(w, X, dims=[[0], [0]]))
cut_cnt = 20
w = torch.tensor([0., 0, 0])
for i in range(cut_cnt):
    w[0] = w_bound * i / cut_cnt
    for j in range(cut_cnt):
        w[1] = (w_bound-w[0]) * j / cut_cnt
        for k in range(cut_cnt):
            w[2] = (w_bound-w[0]-w[1]) * k / cut_cnt
            combination_list.append(torch.tensordot(w, X, dims=[[0], [0]]))
            
x0, x1 = zip(*combination_list)
plt.plot(x0_c, x1_c, '--')
plt.scatter(x0_h, x1_h)
plt.scatter(x0, x1)
plt.axis('off')
plt.show()
    
    