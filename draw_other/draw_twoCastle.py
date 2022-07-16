import os

from ByrdLab.graph import TwoCastle
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
graph = TwoCastle(k=5, byzantine_size=0, seed=40)
graph.show(as_subplot=True)
plt.axis("off")
plt.subplot(1, 2, 2)
graph = TwoCastle(k=7, byzantine_size=4, seed=40)
graph.show(as_subplot=True)
plt.axis("off")

fig = plt.gcf()
SCALE = 1.4
fig.tight_layout()
fig.set_size_inches((SCALE*7, SCALE*3))
pic_path = os.path.join('pic', 'graph_twoCastle' + '.pdf')
plt.savefig(pic_path, format='pdf', bbox_inches='tight')
plt.show()