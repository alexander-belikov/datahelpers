from numpy import zeros, arange, array


def knap(weights, max_weight):
    """
    knap is a solver to a modified knapsack 0/1 problem
    obejctive: take subset from lengths such that sum x_i < max_len
    :param weights: list of ints
    :param max_weight: maximum weight
    :return:
    """
    size = len(weights) + 1
    m = zeros((size, max_weight + 1))
    keep = zeros((size, max_weight + 1))
    m[0] = arange(0, max_weight + 1)
    m[:, max_weight] = max_weight
    for i in range(1, size):
        for l in range(max_weight+1):
            current = weights[i - 1]
            if current < l and m[i - 1, l - current] <= m[i - 1, l]:
                m[i, l] = m[i - 1, l - current]
                keep[i, l] = 1
            else:
                m[i, l] = m[i - 1, l]
                keep[i, l] = 0
    cw = max_weight
    inds = []
    for i in range(size-1, 0, -1):
        if keep[i, cw] == 1:
            inds.append(i-1)
            cw -= weights[i-1]
    return inds


def partition(weights, max_weight):
    """
    partition weights into a list of lists
    each satisfying condition sum x_i < max_len
    :param weights:
    :param max_weight:
    :return:
    """
    ll = array(weights)
    acc = []
    while ll.shape[0] > 0:
        idx = knap(ll, max_weight)
        mask = zeros(ll.shape, dtype=bool)
        mask[idx] = True
        acc.append(list(ll[mask]))
        ll = ll[~mask]
    return acc
