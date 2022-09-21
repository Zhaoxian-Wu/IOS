import random

def local_sample(partition, node, rng_pack=random):
    '''
    node samples an index accroding to the partition
    @partition (list [list]): representing the data partition across all nodes.
        Each element is an interval [a, b], meaning that corresponding node has
        data from index a to b-1
    @node (int): the index of node
    return (int): index of the sample

    [example]: when num = 1
    partition: [[1, 5], [5, 7], [7, 10]]
    if node == 0: return random.randint(1, 4)
    if node == 1: return random.randint(5, 6)
    if node == 2: return random.randint(7, 9)
    '''
    p = partition[node]
    return rng_pack.random.randint(p[0], p[1]-1)

def local_batch_sample(partition, node, counts=1, rng_pack=random):
    '''
    similar to function "local_sample", except this function return a list 
        of sample
    @node (int): the index of node
    @counts (int): the number of sample
    return: a list of sample index
    '''
    p = partition[node]
    return [rng_pack.random.randint(p[0], p[1]-1) for _ in range(counts)]

def local_dataset_iterator(partition, node):
    '''
    @node (int): the index of node
    return: an iterator of local dataset
    '''
    p = partition[node]
    return range(p[0], p[1])
