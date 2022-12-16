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
        super(HorizotalPartition, self).__init__(name, partition)
    def get_subsets(self, dataset):
        return [
            StackedDataSet(features=dataset[p][0], targets=dataset[p][1])
            for i, p in enumerate(self.partition)
        ]
        
    
class EmptyPartition(HorizotalPartition):
    def __init__(self, dataset, node_cnt, rng_pack: RngPackage=RngPackage()):
        partition = [[] for _ in range(node_cnt)]
        super(EmptyPartition, self).__init__('EmptyPartition', partition)
    
    
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
        super(TrivalPartition, self).__init__('TrivalDist', partition)

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
        super(iidPartition, self).__init__('iidPartition', partition)

class SharedData(HorizotalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        partition = [list(range(len(dataset)))] * node_cnt
        super(SharedData, self).__init__('SharedData', partition)
        
class LabelSeperation(HorizotalPartition):
    def __init__(self, dataset, node_cnt, class_cnt, *args, **kw):
        # divide the nodes into `class_cnt` groups
        group_boundary = [(group_idx*node_cnt) // class_cnt 
                       for group_idx in range(class_cnt)]
        # when a data is going to be allocated to `group_idx`-th groups,
        # it'll be allocated to insert_node_ptrs[group_idx]-th node
        # then insert_node_ptrs[group_idx] increases by 1
        insert_node_ptrs = group_boundary.copy()
        group_boundary.append(node_cnt)
        
        partition = [[] for _ in range(node_cnt)]
        for data_idx, (_, label) in enumerate(dataset):
            group_idx = label % node_cnt
            node_idx = insert_node_ptrs[group_idx]
            partition[node_idx].append(data_idx)
            # insert_node_ptrs[group_idx] increases by 1
            if insert_node_ptrs[group_idx] + 1 < group_boundary[group_idx+1]:
                insert_node_ptrs[group_idx] += 1
            else:
                insert_node_ptrs[group_idx] = group_boundary[group_idx]
        super(LabelSeperation, self).__init__('LabelSeperation', partition)
        
class VerticalPartition(Partition):
    def __init__(self, dataset, node_cnt) -> None:
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        seperation = [(i*dataset.feature_dimension) // node_cnt
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
    