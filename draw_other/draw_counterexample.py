import os

from ByrdLab.graph import TwoCastle
import matplotlib.pyplot as plt

k = 3
graph = TwoCastle(k=k, byzantine_size=0, seed=40)
label_dict = {i: '$x_1$' if i<k else '$x_2$' for i in range(2*k)}
graph.show(as_subplot=True, show_label=True, label_dict=label_dict)
plt.axis("off")

fig = plt.gcf()
SCALE = 1.4
fig.set_size_inches((SCALE*3.2, SCALE*3))
pic_path = os.path.join('pic', 'counterexample' + '.pdf')
plt.savefig(pic_path, format='pdf', bbox_inches='tight')
plt.show()