import re
import os
import pickle
import torch
from ByrdLab import DEVICE
 
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(os.getcwd(), __CACHE_DIR__)

def set_cache_path(path):
    global __CACHE_PATH__
    __CACHE_PATH__ = path

def load_file_in_cache(file_name, path_list=[]):
    file_path = get_cache_path(file_name, path_list)
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def dump_file_in_cache(file_name, content, path_list=[]):
    file_path = get_cache_path(file_name, path_list, create_if_not_exist=True)
    with open(file_path, 'wb') as f:
        pickle.dump(content, f)

def dump_model_in_cache(file_name, model, path_list=[]):
    file_path = get_cache_path(file_name, path_list, create_if_not_exist=True)
    with open(file_path, 'wb') as f:
        torch.save(model, f)

def load_model_in_cache(file_name, path_list=[]):
    file_path = get_cache_path(file_name, path_list, create_if_not_exist=True)
    print(file_path)
    with open(file_path, 'rb') as f:
        model = torch.load(f)
    return model

def isfile_in_cache(file_name, path_list=[]):
    file_path = get_cache_path(file_name, path_list)
    return os.path.isfile(file_path)
        
def get_cache_path(file_name, path_list, create_if_not_exist=False):
    file_dir = os.path.join(__CACHE_PATH__, *path_list)
    file_path = os.path.join(file_dir, file_name)
    if create_if_not_exist and not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    return file_path

def move_workspace(path_list, original_ws, target_ws):
    original_dir = os.path.join(__CACHE_PATH__, *path_list, *original_ws)
    target_dir = os.path.join(__CACHE_PATH__, *path_list, *target_ws)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    for file_name in os.listdir(original_dir):
        ori_name = os.path.join(original_dir, file_name)
        if not os.path.isfile(ori_name):
            continue
        target_name = os.path.join(target_dir, file_name)
        os.rename(ori_name, target_name)
    
def change_mark_on_title(path_list, workspace, original_mark, target_mark):
    file_dir = os.path.join(__CACHE_PATH__, *path_list, *workspace)
    pattern_str = '_' + original_mark + '$'
    pattern = re.compile(pattern_str)
    if target_mark != '':
        target_mark = '_' + target_mark
    for file_name in os.listdir(file_dir):
        new_name = pattern.sub(target_mark, file_name)
        ori_name = os.path.join(file_dir, file_name)
        if not os.path.isfile(ori_name):
            continue
        target_name = os.path.join(file_dir, new_name)
        os.rename(ori_name, target_name)