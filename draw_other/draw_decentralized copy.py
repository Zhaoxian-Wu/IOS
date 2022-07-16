import matplotlib.pyplot as plt
from ByrdLab.library.cache_io import load_file_in_cache

def optimal_gap(path, Fmin):
    return [p-Fmin for p in path]

SCALE = 2.0
FONT_SIZE = 12
LINE_STYLE = '-'

files = [
    'DSGD_baseline_RSA_invSqrtLR_lam=0.0001',
    'DSGD_baseline_RSA_invSqrtLR_lam=0.001',
    'DSGD_baseline_RSA_invSqrtLR_lam=0.005',
    'DSGD_baseline_RSA_invSqrtLR_lam=0.02',
    'DSGD_baseline_RSA_invSqrtLR_lam=0.1',
    'DSGD_baseline_RSA_invSqrtLR_lam=1',
]
path_list = [
    'SR_mnist', 'TwoCastle', 'LabelSeperation', 'rsa_tuning'
]

fig, axes_list = plt.subplots(1, 3)
axes_loss, axes_acc, axes_ce = axes_list
for file in files:

    record = load_file_in_cache(file, path_list)
    x_axis = x_axis = [r*record['display_interval']
                       for r in range(record['rounds']+1)]
    
    loss_path = record['loss_path']
    acc_path = record['acc_path']
    consensus_error_path = record['consensus_error_path']
    
    axes_loss.plot(x_axis, loss_path, LINE_STYLE, label=file)
    axes_acc.plot(x_axis, acc_path, LINE_STYLE, label=file)
    axes_ce.plot(x_axis, consensus_error_path, LINE_STYLE, label=file)
    # axes_ce.plot(x_axis[1:], [ce * k for k, ce in enumerate(consensus_error_path)], 'v-', label=file)
    

axes_ce.legend(fontsize=FONT_SIZE)

# loss
axes_loss.set_ylabel(r'$f(x^k)-f(x^*)$', fontsize=FONT_SIZE)
axes_loss.set_xlabel(r'iteration k', fontsize=FONT_SIZE)
axes_loss.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
axes_loss.set_yscale('log')
# accuracy
axes_acc.set_ylabel(r'accuracy', fontsize=FONT_SIZE)
axes_acc.set_xlabel(r'iteration k', fontsize=FONT_SIZE)
axes_loss.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
# consensus error
axes_ce.set_ylabel(r'consensus error', fontsize=FONT_SIZE)
axes_ce.set_xlabel(r'iteration k', fontsize=FONT_SIZE)
axes_loss.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
axes_loss.set_yscale('log')

fig.set_size_inches((SCALE*5, SCALE*+1))
plt.subplots_adjust(hspace=0.35, wspace=0.35)
# plt.savefig('./adjust.pdf', format='pdf', bbox_inches='tight')
plt.show()