from numpy import zeros, arange, array, concatenate, argsort
from scipy.stats import ks_2samp

#TODO what if there exist a weight >= max_weight?
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


def bin_packing_ffd_mod(weights, pdfs, max_size, violation_level=0., distance_func=ks_2samp):
    """

    :param weights:
    :param pdfs:
    :param max_size:
    :param violation_level:
    :param distance_func:
    :return:
    """
    sample0 = concatenate(pdfs)
    inds_sorted = argsort(weights)[::-1]
    inds2 = list(inds_sorted)
    weights2 = list(weights[inds_sorted])
    pdfs2 = list(pdfs[inds_sorted])
    bins = [[]]
    r_pdfs = [[]]
    ind_cur_bin = 0
    if weights2[0] > max_size:
        return False, []
    improves_pdf = True

    lower_bound_bins_number = int(round(sum(weights) / max_size + 0.5))
    bins = [[x] for x in weights2[:lower_bound_bins_number]]
    r_pdfs = [[x] for x in pdfs2[:lower_bound_bins_number]]
    indices = [[i] for i in inds2[:lower_bound_bins_number]]

    weights2 = weights2[lower_bound_bins_number:]
    pdfs2 = pdfs2[lower_bound_bins_number:]
    inds2 = inds2[lower_bound_bins_number:]

    while weights2:
        dispatched = False
        cnt = 1
        ind_cur_ssample = 0
        while not dispatched:
            cur_bin = bins[ind_cur_bin]
            cur_pdf_bin = r_pdfs[ind_cur_bin]
            cur_ind_bin = indices[ind_cur_bin]

            if cur_pdf_bin:
                ks_cur = distance_func(concatenate(cur_pdf_bin), sample0)
                ks_cur2 = distance_func(concatenate(cur_pdf_bin + [pdfs2[ind_cur_ssample]]), sample0)
                # print(sample0.shape, ks_cur, ks_cur2)
                improves_pdf = (ks_cur2[0] < ks_cur[1] + violation_level)
            if max_size - sum(cur_bin) >= weights2[ind_cur_ssample] and improves_pdf:
                cur_bin.append(weights2.pop(ind_cur_ssample))
                cur_pdf_bin.append(pdfs2.pop(ind_cur_ssample))
                cur_ind_bin.append(inds2.pop(ind_cur_ssample))
                dispatched = True
            elif cnt < len(bins):
                cnt += 1
                ind_cur_bin = (ind_cur_bin + 1) % len(bins)
            else:
                if ind_cur_ssample < len(pdfs2) - 1:
                    ind_cur_ssample += 1
                else:
                    cur_bin = []
                    cur_pdf_bin = []
                    cur_ind_bin = []
                    ind_cur_ssample = 0
                    cur_bin.append(weights2.pop(ind_cur_ssample))
                    cur_pdf_bin.append(pdfs2.pop(ind_cur_ssample))
                    cur_ind_bin.append(inds2.pop(ind_cur_ssample))
                    bins.insert(ind_cur_bin, cur_bin)
                    r_pdfs.insert(ind_cur_bin, cur_pdf_bin)
                    indices.insert(ind_cur_bin, cur_ind_bin)
                    dispatched = True
    return True, bins, r_pdfs, indices


def ks_2samp_multi_dim(sample_a, sample_b):
    # p_val is not additive
    s = 0
    for x, y in zip(sample_a.T, sample_b.T):
        r = ks_2samp(x, y)
        s += r[0]
    p_val = 1.0
    return s, p_val


def partition_dict(dict_items, ind_start, ind_end, max_size):
    """

    :param dict_items: a dict of numpy arrays with equal first dimension
    :param ind_start: index of first ar
    :param ind_end:
    :param max_size:
    :return:
    """
    order_keys = list(dict_items.keys())
    ordered_weights = array([dict_items[k].shape[1] for k in order_keys])
    ordered_data = array([dict_items[k][ind_start:ind_end].T for k in order_keys])
    b, lens_mod, pdfs_mod, inds = bin_packing_ffd_mod(ordered_weights, ordered_data,
                                                      max_size, 0.01, ks_2samp_multi_dim)
    split_keys = [[order_keys[j] for j in ind_batch] for ind_batch in inds]
    return split_keys
