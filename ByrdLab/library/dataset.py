import ByrdLab
from ByrdLab.library.RandomNumberGenerator import RngPackage, torch_rng
import random
import re
import os
import torch

from ByrdLab import FEATURE_TYPE, CLASS_TYPE, VALUE_TYPE
from .cache_io import isfile_in_cache, load_file_in_cache, dump_file_in_cache

def load_SVM_dataset(dataset_file, set_size, dimension,
                     finding_type, multi_class=False):
    X = torch.zeros((set_size, dimension), dtype=FEATURE_TYPE)
    Y = torch.zeros((set_size), dtype=CLASS_TYPE)

    with open(dataset_file, 'r') as f:
        for line, vector in enumerate(f):
            cat, data = vector.split(' ', 1)
            if multi_class:
                Y[line] = cat
            else:
                Y[line] = 1 if cat == finding_type else 0
            for piece in data.strip().split(' '):
                match = re.search(r'(\S+):(\S+)', piece)
                # feature in dataset starts from 1, so we substract 1 here
                feature = int(match.group(1)) - 1
                value = float(match.group(2))
                # record features
                X[line][feature] = value
    return X, Y

class Dataset():
    def __init__(self, name, features, targets,
                 val_features=None, val_targets=None):
        assert len(features) == len(targets)
        set_size = len(features)
        self.features = features
        self.targets = targets
        self.val_features = val_features
        self.val_targets = val_targets

        # random shuffling
        self.__RR = False
        self.__order = list(range(set_size))

        self.name = name
        self.set_size = set_size
        self.__COUNT = None
        if len(features) != 0:
            self.feature_dimension = features[0].nelement()
            self.feature_size = features[0].size()
        else:
            self.feature_dimension = 0
            self.feature_size = 0
    def get_count(self):
        if self.__COUNT is None:
            count = {}
            for target_tensor in self.targets:
                target = target_tensor.item()
                if target in count.keys():
                    count[target] += 1
                else:
                    count[target] = 1
            self.__COUNT = count
        return self.__COUNT
        
    def randomReshuffle(self, on):
        self.__RR = on
        if on:
            random.shuffle(self.__order)
    def __getitem__(self, index):
        if self.__RR:
            i = self.__order[index]
            return self.features[i], self.targets[i]
        else:
            return self.features[index], self.targets[index]
    def __len__(self):
        return self.set_size
    def subset(self, indexes, name=''):
        if name == '':
            name = self.name + '_subset'
        return Dataset(self.name, self.features[indexes], self.targets[indexes])
    def get_val_set(self):
        if self.val_features is None or self.val_targets is None:
            self.val_features = self.features
            self.val_targets = self.targets
        return Dataset(self.name+'_val', self.val_features, self.val_targets)
    
class SVM_dataSet(Dataset):
    def __init__(self, set_size, dimension, dataset_name, dataset_file,
                 finding_type=0, multi_class=False):
        # try to load processed dataset from cache
        dtype_str = str(ByrdLab.FEATURE_TYPE).replace('torch.', '')
        cache_file_name = f'data_cache_{dataset_name}_{dtype_str}'
        if isfile_in_cache(cache_file_name):
            cache = load_file_in_cache(cache_file_name)
            X, Y = cache['X'], cache['Y']
        else:
            X, Y = load_SVM_dataset(dataset_file, set_size, dimension, 
                                    finding_type, multi_class)
            cache = {
                'X': X,
                'Y': Y,
            }
            dump_file_in_cache(cache_file_name, cache)
            
        super(SVM_dataSet, self).__init__(name=dataset_name, features=X, 
                                          targets=Y)
        self._finding_type = finding_type

__DATASET_PATH__ = os.path.join(
    os.getcwd(), 'dataset'
)

class ijcnn(SVM_dataSet):
    def __init__(self):
        super(ijcnn, self).__init__(
            set_size = 49990,
            dimension = 22,
            finding_type = '1',
            dataset_name = 'ijcnn1',
            dataset_file = os.path.join(__DATASET_PATH__, 'ijcnn1')
        )

class covtype(SVM_dataSet):
    def __init__(self):
        super(covtype, self).__init__(
            set_size = 581012,
            dimension = 54,
            finding_type = '1',
            dataset_name = 'covtype',
            dataset_file = os.path.join(__DATASET_PATH__, 
                'covtype.libsvm.binary.scale'
            )
        )
        
class torchDataset(Dataset):
    def __init__(self, name, torch_train_set, torch_val_set):
        self.torch_train_set = torch_train_set
        self.torch_val_set = torch_val_set
        dtype_str = str(ByrdLab.FEATURE_TYPE).replace('torch.', '')
        cache_file_name = f'data_cache_{name}_{dtype_str}'
        if isfile_in_cache(cache_file_name):
            cache = load_file_in_cache(cache_file_name)
            train_features = cache['train_features']
            train_targets = cache['train_targets']
            val_features = cache['val_features']
            val_targets = cache['val_targets']
        else:
            train_features = \
                torch.stack([feature for feature, _ in torch_train_set], axis=0)
            val_features = \
                torch.stack([feature for feature, _ in torch_val_set], axis=0)
            # feature_size = torch_train_set[0][0].size()
            # dataset_size = len(torch_train_set)
            # train_features = torch.empty(dataset_size, *feature_size)
            train_targets = torch_train_set.targets
            val_targets = torch_val_set.targets
            # for i, (feature, _) in enumerate(torch_train_set):
            #     train_features[i].copy_(feature)
            cache = {
                'train_features': train_features,
                'train_targets': train_targets,
                'val_features': val_features,
                'val_targets': val_targets,
            }
            dump_file_in_cache(cache_file_name, cache)
        
        self.num_classes = len(torch_train_set.classes)
        super().__init__(name, train_features, torch_train_set.targets,
                         val_features, torch_val_set.targets)
    def subset(self, indexes, name=''):
        s = super().subset(indexes, name)
        s.num_classes = self.num_classes
        return s
        
class mnist(torchDataset):
    def __init__(self):
        from torchvision import transforms
        from torchvision.datasets import MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=ByrdLab.FEATURE_TYPE),
            # transforms.Normalize(mean=[0.1307],std=[0.3081])
            transforms.Normalize(mean=[0.5],std=[0.5])
            # transforms.Lambda(lambda x: x / 255)
        ])
        root = 'dataset'
        torch_train_dataset = MNIST(root=root, train=True,
                                    transform=transform, download=False)
        torch_val_dataset = MNIST(root=root, train=False,
                                  transform=transform, download=False)
        super().__init__('mnist', torch_train_dataset, torch_val_dataset)

class LogisticRegressionToySet(Dataset):
    def __init__(self, set_size, dimension, noise=0.1, w_star=None,
                 fix_seed=False, seed=20):
        generator = torch_rng(seed) if fix_seed else None
        X = torch.randn((set_size, dimension), dtype=FEATURE_TYPE,
                        generator=generator)
        Y = torch.zeros((set_size), dtype=CLASS_TYPE)
        if w_star == None:
            w_star = torch.randn(dimension + 1, dtype=FEATURE_TYPE,
                                 generator=generator)
        assert w_star.size() == torch.Size([dimension + 1])
        self.w_star = w_star

        for i, x in enumerate(X):
            if self.w_star[:-1].dot(x) + self.w_star[-1] > 0:
                Y[i] = 1
            else:
                Y[i] = 0

        # add noise
        X.add_(torch.randn_like(X), alpha=noise)
        
        super(LogisticRegressionToySet, self).__init__(name='ToySet',
                                                       features=X, targets=Y)

    
class DistributedDataSets():
    def __init__(self, dataset, partition_cls, nodes, honest_nodes,
                 rng_pack: RngPackage=RngPackage()):
        self.dataset = dataset
        
        self.partition = partition_cls(dataset, len(honest_nodes),
                                       rng_pack=rng_pack)
        honest_subsets = self.partition.get_subsets(dataset)
        
        # allocate the partitions to all nodes
        next_pointer = 0
        self.subsets = [[] for _ in nodes]
        for node in nodes:
            if node in honest_nodes:
                self.subsets[node] = honest_subsets[next_pointer]
                next_pointer += 1
    def __getitem__(self, index):
        return self.subsets[index]
    def entire_set(self):
        return self.dataset
    
    
class EmptySet(Dataset):
    def __init__(self):
        super().__init__(name='EmptySet', 
                         features=torch.zeros(0), targets=torch.zeros(0), 
                         val_features=torch.zeros(0), val_targets=torch.zeros(0))
    