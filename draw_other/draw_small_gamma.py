# %%
import matplotlib.pyplot as plt
import os
import pickle
import math

dataSetConfigs = [
    {
        'name': 'ijcnn1',

        'dataSet' : 'ijcnn1',
        'dataSetSize': 49990,
        'maxFeature': 22,
        'findingType': '1',

        'honestNodeSize': 50,
        'byzantineNodeSize': 20,

        'rounds': 10,
        'displayInterval': 4000,
    },
    # {
    #     'name': 'covtype',

    #     'dataSet' : 'covtype.libsvm.binary.scale',
    #     'dataSetSize': 581012,
    #     'maxFeature': 54,
    #     'findingType': '1',

    #     'honestNodeSize': 50,
    #     'byzantineNodeSize': 20,

    #     'rounds': 10,
    #     'displayInterval': 4000,
    # }
]
dataSetConfig = dataSetConfigs[0]

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
__CACHE_DIR__ = 'record'
__CATCH_PATH__ = os.path.join(__FILE_DIR__, os.path.pardir, __CACHE_DIR__)

def logAxis(path, Fmin):
    return [p-Fmin for p in path]

SCALE = 1.5
FONT_SIZE = 12

gammas = [0.01, 0.001, 0.0005, 0.0001]

record_prefix = dataSetConfig['name'] + '_'
CACHE_DIR = os.path.join(__CATCH_PATH__, dataSetConfig['name'])
for gamma in gammas:
    
    with open(CACHE_DIR + '_Fmin', 'rb') as f:
        obj = pickle.load(f)
        Fmin, w_min = obj['Fmin'], obj['w_min']

    
    file_name = CACHE_DIR + f'_SGD_sign_flipping_geometric_median_gamma={gamma}'
    with open(file_name, 'rb') as f:
        x_axis = list([r*record['display_interval'] for r in range(record['rounds']+1)])
        record = pickle.load(f)
        path = record['path']
        path = logAxis(path, Fmin)
        plt.plot(x_axis, path, 'v-', label=rf'$\gamma={gamma}$')
    
plt.xticks(size = FONT_SIZE)
plt.yticks(size = FONT_SIZE)
plt.yscale('log')
plt.ylabel(r'$f(x^k)-f(x^*)$', fontsize=FONT_SIZE)
plt.xlabel(r'iteration k / ${}$'.format(dataSetConfig['displayInterval']),
            fontsize=FONT_SIZE)
# labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
# [label.set_fontsize(FONT_SIZE*3) for label in labels]

# 图例
plt.legend(fontsize=FONT_SIZE)

plt.gcf().set_size_inches((SCALE*6, SCALE*4))
plt.savefig('./pic/small_gamma.pdf', format='pdf', bbox_inches='tight')
plt.show()
