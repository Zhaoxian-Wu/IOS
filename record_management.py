from ByrdLab.library.cache_io import move_workspace
path_list = ['SR_mnist', 'TwoCastle', 'LabelSeperation']
original_ws = []
target_ws = ['temp']
move_workspace(path_list, original_ws, target_ws)

from ByrdLab.library.cache_io import change_mark_on_title
path_list = ['SR_mnist', 'TwoCastle', 'LabelSeperation']
workspace = []
ori = 'lr=0.9'
tar = ''
change_mark_on_title(path_list, workspace, ori, tar)