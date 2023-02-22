from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab.library.dataset import DataPackage, StackedDataSet

class Partition():
    def __init__(self, name, partition, rng_pack: RngPackage=RngPackage()):
        self.name = name
        self.partition = partition
        self.rng_pack = rng_pack
    def get_subsets(self, dataset):
        '''
        return all subsets of dataset
        ---------------------------------------
        TODO: the partition of data depends on the specific structure
              of dataset.
              In the version, dataset has the structure that all features
              and targets are stacked in tensors. For other datasets with
              different structures, another type of `get_subsets` shoule
              be implemented.
        '''
        raise NotImplementedError
    def __getitem__(self, i):
        return self.partition[i]
    def __len__(self):
        return len(self.partition)
    
    
class HorizotalPartition(Partition):
    def __init__(self, name, partition):
        self.partition = partition
        super().__init__(name, partition)
    def get_subsets(self, dataset):
        return [
            StackedDataSet(features=dataset[p][0], targets=dataset[p][1])
            for i, p in enumerate(self.partition)
        ]
        
    
class EmptyPartition(HorizotalPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        partition = [[] for _ in range(node_cnt)]
        super().__init__('EmptyPartition', partition)
    
    
class TrivalPartition(HorizotalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        seperation = [(node*len(dataset)) // node_cnt 
                      for node in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [list(range(seperation[i], seperation[i+1]))
                                for i in range(node_cnt)]
        super().__init__('TrivalDist', partition)

class iidPartition(HorizotalPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()) -> None:
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        indexes = list(range(len(dataset)))
        rng_pack.random.shuffle(indexes)
        sep = [(i*len(dataset)) // node_cnt for i in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [[indexes[i] for i in range(sep[node], sep[node+1])]
                                for node in range(node_cnt)]
        super().__init__('iidPartition', partition)

class SharedData(HorizotalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        partition = [list(range(len(dataset)))] * node_cnt
        super().__init__('SharedData', partition)
        
class LabelSeperation(HorizotalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw):
        self.class_set = set([label.item() for _, label in dataset])
        self.class_cnt = len(self.class_set)
        self.node_cnt = node_cnt
        self.dataset = dataset
        # deal with the situation that class idx don't
        # locate in consecutive integers starting from zeros
        self.class_idx_dict = {
            label: idx for idx, label in enumerate(self.class_set)}
        
        if self.class_cnt < node_cnt:
            partition = self.partition_with_adaquate_nodes()
        else:
            partition = self.partition_with_adaquate_classes()
        super().__init__('LabelSeperation', partition)
        
    def partition_with_adaquate_classes(self):
        '''
        class_cnt >= node_cnt
        some nodes possess several classes
        '''
        partition = [[] for _ in range(self.node_cnt)]
        for data_idx, (_, label) in enumerate(self.dataset):
            node_idx = self.class_idx_dict[label.item()] % self.node_cnt
            partition[node_idx].append(data_idx)
        return partition
    
    def partition_with_adaquate_nodes(self):
        '''
        class_cnt < node_cnt
        some classes are allocated on different workers
        '''
        class_cnt = self.class_cnt
        node_cnt = self.node_cnt
        dataset = self.dataset
        
        # divide the nodes into `class_cnt` groups
        group_boundary = [(group_idx*node_cnt) // class_cnt 
                            for group_idx in range(class_cnt)]
        # when a data is going to be allocated to `group_idx`-th groups,
        # it'll be allocated to `insert_node_ptrs[group_idx]`-th node
        # then `insert_node_ptrs[group_idx]` increases by 1
        insert_node_ptrs = group_boundary.copy()
        group_boundary.append(node_cnt)
        # [e.g] 
        # class_cnt = 5
        # node_cnt = 8
        # group_boundary = [0, 1, 3, 4, 6, 8]
        # divide 8 nodes into 5 groups by
        # 0 | 1 | 2 3 | 4 5 | 6 7 |
        # where the vertical line represent the corresponding `group_boundary`
        # this means
        # class 0 on worker 0
        # class 1 on worker 1
        # class 2 on worker 2, 3
        # class 3 on worker 4, 5
        # class 4 on worker 6, 7
        
        partition = [[] for _ in range(node_cnt)]
        for data_idx, (_, label) in enumerate(dataset):
            # determine which group the data belongs to
            group_idx = self.class_idx_dict[label.item()]
            node_idx = insert_node_ptrs[group_idx]
            partition[node_idx].append(data_idx)
            # `insert_node_ptrs[group_idx]` increases by 1
            if insert_node_ptrs[group_idx] + 1 < group_boundary[group_idx+1]:
                insert_node_ptrs[group_idx] += 1
            else:
                insert_node_ptrs[group_idx] = group_boundary[group_idx]
        return partition
        
        
class VerticalPartition(Partition):
    def __init__(self, dataset: StackedDataSet, 
                 node_cnt: int, *args, **kw) -> None:
        feature_dimension = dataset.feature_dimension
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        seperation = [(i*feature_dimension) // node_cnt
                      for i in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [list(range(seperation[i], seperation[i+1]))
                                for i in range(node_cnt)]
        super().__init__('TrivalDist', partition)
    def get_subsets(self, dataset):
        return [
            StackedDataSet(features=dataset.features[:, p],
                           targets=dataset.targets)
            for i, p in enumerate(self.partition)
        ]
    