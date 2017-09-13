from numpy import zeros, arange, array, concatenate, \
    argsort, abs, sum, argmin, std, mean, tile, argwhere, where, ceil
from scipy.stats import ks_2samp
from copy import deepcopy


def knap(weights, max_weight):
    """
    knap is a solver to a modified knapsack 0/1 problem via DP
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
    solve bin packing via knapsack
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
    bin packing according to weights and pdf distance
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
    ind_cur_bin = 0
    if weights2[0] > max_size:
        raise ValueError('Max item weight is greater than proposed bin cap')
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
                improves_pdf = (ks_cur2[0] < ks_cur[0] + violation_level)
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
    """
    compute distance between two k-dim samples
    :param sample_a:
    :param sample_b:
    :return:
    """
    # p_val is not additive
    s = 0
    for x, y in zip(sample_a.T, sample_b.T):
        r = ks_2samp(x, y)
        s += r[0]
    p_val = r[1]
    return s, p_val


def partition_dict_to_subsamples(dict_items, number_of_bins):
    """

    :param dict_items: a dict of numpy arrays with equal first dimension
    :param ind_start: index of first ar
    :param ind_end:
    :param number_of_bins: tentative number of bins
    :return:
    """
    order_keys = list(dict_items.keys())
    ordered_data = [dict_items[k] for k in order_keys]
    iis = bin_packing_mean(ordered_data, number_of_bins, distance_func=ks_2samp_multi_dim)
    iis2 = reshuffle_bins(iis, ordered_data, 0.5, ks_2samp_multi_dim)
    split_keys = [[order_keys[j] for j in ind_batch] for ind_batch in iis2]
    return split_keys


def try_moving_element(item_a, item_b, mean_phi_over_weights, sample0,
                       epsilon=0.5, distance_func=ks_2samp):
    """
    try moving elements from item_b to item_a to decrease pdf distance and make weight metrics more similar
    :param item_a: weights_a, pdf_a
    :param item_b: weights_b, pdf_b
    :param mean_phi_over_weights:
    :param sample0:
    :param epsilon:
    :param distance_func:
    :return:

    given
    weights_a, pdf_a = item_a
    weights_b, pdf_b = item_b
    take one kth element from weights_b, pdf_b in such a way that
    la'*sa' is closer mean_phi_over_weights and
    distances rho(pdf_a', sample0) and rho(pdf_b', sample0) are improved

    epsilon controls how much importance is given to weight based metric vs pdf metric
    epsilon = 1 only pdf metric; epsilon = 0 only weight based metric

    """

    weight_a, pdf_a = item_a
    weight_b, pdf_b = item_b
    if not (len(weight_a) == len(pdf_a) and len(weight_b) == len(pdf_b)):
        raise ValueError('cardinality of indices, weights and pdfs are not equal')
    len_a, len_b = len(weight_a), len(weight_b)
    sum_a, sum_b = sum(weight_a), sum(weight_b)
    pi_a, pi_b = len_a * sum_a, len_b * sum_b
    da0 = distance_func(concatenate(pdf_a), sample0)[0]
    db0 = distance_func(concatenate(pdf_b), sample0)[0]
    delta_vector = array(weight_b)
    delta_a = abs(pi_a - mean_phi_over_weights)
    delta_b = abs(pi_b - mean_phi_over_weights)
    diff_a = abs(pi_a + sum_a + (len_a+1)*delta_vector - mean_phi_over_weights)/delta_a
    diff_b = abs(pi_b - sum_b - (len_b-1)*delta_vector - mean_phi_over_weights)/delta_b
    # conversion to a flat list
    indices_b = argwhere(((1. - diff_a) > 0) & ((1. - diff_b) > 0)).flatten().tolist()
    if indices_b:
        pi_dist = []
        pdf_dist = []
        for jb in indices_b:
            da, _ = distance_func(concatenate([pdf_b[jb]] + pdf_a), sample0)
            db, _ = distance_func(concatenate([pdf_b[k] for k in range(len_b) if k != jb]), sample0)
            pi_dist.append((diff_a[jb], diff_b[jb]))
            pdf_dist.append((da/da0, db/db0))
        pi_dist_arr = (array(pi_dist)**2).sum(axis=1)
        pdf_dist_arr = array(pdf_dist)
        pdf_dist_arr /= abs(pdf_dist_arr.max(axis=0))
        pdf_dist_arr = abs(pdf_dist_arr)
        pdf_dist_conv = (pdf_dist_arr ** 2).sum(axis=1)
        distances_normed = (epsilon * pdf_dist_conv + (1. - epsilon) * pi_dist_arr ** 2) ** 0.5
        j_best = argmin(distances_normed)
        swap_flag = True
    else:
        swap_flag = False
        j_best = -1

    return swap_flag, (None, j_best)


def try_swapping_elements(item_a, item_b, mean_phi_over_weights, sample0,
                          epsilon=0.5, distance_func=ks_2samp):
    """
    try swapping elements between item_a and item_b to decrease pdf distance and make weight metrics more similar

    :param item_a: weight_a, pdf_a
    :param item_b: weight_b, pdf_b
    :param mean_phi_over_weights:
    :param sample0:
    :param epsilon:
    :param distance_func:
    :return: sawp_flag, (index of item_a element, index of item_b element)
    epsilon controls how much importance is given to weight based metric vs pdf metric
    epsilon = 1 only pdf metric; epsilon = 0 only weight based metric
    """

    weight_a, pdf_a = item_a
    weight_b, pdf_b = item_b
    if not (len(weight_a) == len(pdf_a) and len(weight_b) == len(pdf_b)):
        raise ValueError('cardinality of indices, weights and pdfs are not equal')
    len_a, len_b = len(weight_a), len(weight_b)
    sum_a, sum_b = sum(weight_a), sum(weight_b)
    pi_a, pi_b = len_a * sum_a, len_b * sum_b
    da0 = distance_func(concatenate(pdf_a), sample0)[0]
    db0 = distance_func(concatenate(pdf_b), sample0)[0]

    # a_ij = w^a_i - w^b_j
    delta_matrix = tile(weight_a, (len(weight_b), 1)).T - array(weight_b)
    delta_a = abs(pi_a - mean_phi_over_weights)
    delta_b = abs(pi_b - mean_phi_over_weights)
    diff_a = abs(pi_a - len_a * delta_matrix - mean_phi_over_weights)/delta_a
    diff_b = abs(pi_b + len_b * delta_matrix - mean_phi_over_weights)/delta_b
    pairs = argwhere(((1. - diff_a) > 0) & ((1. - diff_b) > 0))
    if pairs.any():
        pi_dist = []
        pdf_dist = []
        for ja, jb in pairs:
            pdf_a_, pdf_b_ = deepcopy(pdf_a), deepcopy(pdf_b)
            pdf_a_[ja] = pdf_b[jb]
            pdf_b_[jb] = pdf_a[ja]
            da = distance_func(concatenate(pdf_a_), sample0)[0]
            db = distance_func(concatenate(pdf_b_), sample0)[0]
            pi_dist.append((diff_a[ja, jb], diff_b[ja, jb]))
            pdf_dist.append((da/da0, db/db0))
        pi_dist_arr = (array(pi_dist)**2).sum(axis=1)
        pdf_dist_arr = array(pdf_dist)
        pdf_dist_arr /= abs(pdf_dist_arr.max(axis=0))
        pdf_dist_arr = abs(pdf_dist_arr)
        pdf_dist_conv = (pdf_dist_arr ** 2).sum(axis=1)
        distances_normed = (epsilon * pdf_dist_conv + (1. - epsilon) * pi_dist_arr ** 2) ** 0.5
        ja, jb = pairs[argmin(distances_normed)]
        swap_flag = True
    else:
        swap_flag = False
        ja, jb = -1, -1
    return swap_flag, (ja, jb)


def manage_lists(partition_inds, weights, pdfs, sample0, mask_func,
                 foo=try_moving_element, epsilon=0.5, distance_func=ks_2samp_multi_dim):
    """
    arrange items from partition, to apply foo (to move or swap elements from items)
    :param partition_inds:
    :param weights:
    :param pdfs:
    :param sample0:
    :param mask_func:
    :param foo:
    :param epsilon:
    :param distance_func:
    :return:
    """

    bins = [[weights[j] for j in ind_batch] for ind_batch in partition_inds]
    pdf_bins = [[pdfs[j] for j in ind_batch] for ind_batch in partition_inds]
    ls = list(map(len, bins))
    ss = list(map(sum, bins))
    ps = list(map(lambda x: x[0] * x[1], zip(ls, ss)))
    mean_ps = mean(ps)

    ps_sorted_inds = argsort(ps)
    ls_sorted = array(ls)[ps_sorted_inds]
    ps_sorted = array(ps)[ps_sorted_inds]
    ls_matrix = tile(ls_sorted, (len(ls), 1)).T - ls_sorted
    pairs = argwhere(mask_func(ls_matrix))[::-1, ::-1]
    diffs = [ps_sorted[y] - ps_sorted[x] for x, y in pairs]
    ind_diff_sort = argsort(diffs)
    pps = [list(pairs[j]) for j in ind_diff_sort]

    # make sure first index is a lower number of items in the bin
    while pps:
        ia, ib = pps.pop()
        index_a, index_b = ps_sorted_inds[ia], ps_sorted_inds[ib]
        bin_a, bin_b = bins[index_a], bins[index_b]
        pdf_a, pdf_b = pdf_bins[index_a], pdf_bins[index_b]
        accepted, (ja, jb) = foo((bin_a, pdf_a), (bin_b, pdf_b), mean_ps, sample0, epsilon, distance_func)
        if accepted:
            partition_ind_a, partition_ind_b = list(partition_inds[index_a]), list(partition_inds[index_b])
            if ja:
                partition_ind_a.pop(ja)
            if jb:
                partition_ind_b.pop(jb)
            if ja:
                partition_ind_b.append(partition_inds[index_a][ja])
            if jb:
                partition_ind_a.append(partition_inds[index_b][jb])
            pps = [pp for pp in pps if pp[0] != ia and pp[1] != ib and pp[0] != ib and pp[1] != ia]
            partition_inds[index_a] = partition_ind_a
            partition_inds[index_b] = partition_ind_b
    return partition_inds


def reshuffle_bins(partition_indices, pdfs, epsilon=0.5, distance_func=ks_2samp):
    """
    try to rearrange items given partion (move and swap)
    :param partition_indices:
    :param pdfs:
    :param epsilon:
    :param distance_func:
    :return:
    """
    partition_indices_new = deepcopy(partition_indices)
    sample0 = concatenate(pdfs)
    weights = [d.shape[0] for d in pdfs]

    report = check_packing(partition_indices_new, weights, pdfs)
    print(report, '; std/mean = {0}'.format(report[1]/report[0]))

    partition_indices_new = manage_lists(partition_indices_new, weights, pdfs, sample0,
                                         lambda x: x >= 1, try_moving_element, epsilon, distance_func)
    report = check_packing(partition_indices_new, weights, pdfs)
    print(report, '; std/mean = {0}'.format(report[1]/report[0]))

    partition_indices_new = manage_lists(partition_indices_new, weights, pdfs, sample0,
                                         lambda x: x < 1, try_swapping_elements, epsilon, distance_func)
    report = check_packing(partition_indices_new, weights, pdfs)
    print(report, '; std/mean = {0}'.format(report[1]/report[0]))

    return partition_indices_new


def bin_packing_mean(pdfs_input, number_bins, distance_func=ks_2samp):
    """
    partition list of numpy arrays pdfs_input into a list of lists

    :param pdfs_input:
    :param number_bins: tentative number of bins
    :param distance_func: sample distance func
    :return:
    """

    # descending order in size and in std
    inds = [x[0] for x in sorted(enumerate(pdfs_input),
                                 key=lambda y: (y[1].shape[0], y[1].std()), reverse=True)]

    weights = [pdfs_input[ii].shape[0] for ii in inds]
    pdfs = [pdfs_input[ii] for ii in inds]

    w_mean0 = mean(weights)
    sample0 = concatenate(pdfs_input)
    items_per_bin = int(ceil(len(weights) / number_bins))
    bin_capacity = (items_per_bin * w_mean0)
    bin_product = bin_capacity * items_per_bin

    if max(weights) > bin_capacity:
        raise ValueError('Max item weight is greater than proposed bin cap')
    # populate each bin with a largest available element
    bins = [[x] for x in weights[:number_bins]]
    indices = [[i] for i in inds[:number_bins]]
    pdf_bins = [[i] for i in pdfs[:number_bins]]

    indices_output = []
    weights = weights[number_bins:]
    inds = inds[number_bins:]
    pdfs = pdfs[number_bins:]

    diffs = [x - y for x, y in zip(weights[:-1], weights[1:])]
    bbs = [0] + [j + 1 for j in range(len(diffs)) if diffs[j] != 0]
    pdfs2 = [pdfs[bbs[i]:bbs[i + 1]] for i in range(len(bbs) - 1)] + [pdfs[bbs[-1]:]]
    inds2 = [inds[bbs[i]:bbs[i + 1]] for i in range(len(bbs) - 1)] + [inds[bbs[-1]:]]
    weights_uni = [weights[bbs[i]] for i in range(len(bbs))]
    j_cur_bin = 0
    k_cur_weight = 0

    state = -1, -1, -1, -1
    while weights_uni:
        ind_bin = indices[j_cur_bin]
        wei_bin = bins[j_cur_bin]
        pdf_bin = pdf_bins[j_cur_bin]
        ind_strata = inds2[k_cur_weight]
        wei_strata = weights_uni[k_cur_weight]
        pdf_strata = pdfs2[k_cur_weight]

        pi_tentative = (sum(wei_bin) + wei_strata) * (len(wei_bin) + 1)
        pi_tentative_min = (sum(wei_bin) + min(weights_uni)) * (len(wei_bin) + 1)
        if pi_tentative < bin_product:
            dists = []
            for pdf in pdf_strata:
                bin_pdf_dist = distance_func(concatenate(pdf_bin), sample0)[0]
                bin_pdf_dist_new = distance_func(concatenate(pdf_bin + [pdf]), sample0)[0]
                diff_pdf = bin_pdf_dist_new / bin_pdf_dist
                dists.append(diff_pdf)
            j_best = argmin(array(dists) ** 2)
            ind_bin.append(ind_strata.pop(j_best))
            pdf_bin.append(pdf_strata.pop(j_best))
            wei_bin.append(wei_strata)
            accepted = True
            state = j_cur_bin, k_cur_weight, len(bins), len(weights_uni)
        else:
            accepted = False
        if not accepted and state == (j_cur_bin, k_cur_weight, len(bins), len(weights_uni)):
            print('Loop detected')
            indices.append([ind_strata.pop()])
            pdf_bins.append([pdf_strata.pop()])
            bins.append([wei_strata])
        if not ind_strata:
            inds2.pop(k_cur_weight)
            pdfs2.pop(k_cur_weight)
            weights_uni.pop(k_cur_weight)
        if pi_tentative_min > bin_product:
            indices_output.append(indices.pop(j_cur_bin))
            pdf_bins.pop(j_cur_bin)
            bins.pop(j_cur_bin)
        if weights_uni:
            k_cur_weight = (k_cur_weight + 1) % len(weights_uni)
        if bins:
            j_cur_bin = (j_cur_bin + 1) % len(bins)
    indices_output.extend(indices)
    return indices_output


def check_packing(list_indices, weights, pdfs, distance_func=ks_2samp_multi_dim):
    """
    return stats of the partition
    :param list_indices:
    :param weights:
    :param pdfs:
    :param distance_func:
    :return:
    """
    if not (sum(list(map(len, list_indices))) == len(weights) and len(weights) == len(pdfs)):
        raise ValueError('cardinality of indices, weights and pdfs are not equal')

    sample0 = concatenate(pdfs)
    bins = [[weights[j] for j in ind_batch] for ind_batch in list_indices]
    pdf_bins = [[pdfs[j] for j in ind_batch] for ind_batch in list_indices]

    ls = list(map(len, bins))
    ss = list(map(sum, bins))
    ps = list(map(lambda x: x[0] * x[1], zip(ls, ss)))
    mean_ps = mean(ps)
    std_ps = std(ps)
    rhos = list(map(lambda x: distance_func(concatenate(x), sample0)[0], pdf_bins))
    return mean_ps, std_ps, min(rhos), max(rhos)


# old material beyond this point

def partition_dict(dict_items, max_size, how='len'):
    """
    return keys of partition of dict of numpy array into groups of at most max size

    :param dict_items: a dict of numpy arrays with equal first dimension
    :param ind_start: index of first ar
    :param ind_end:
    :param max_size:
    :param how
    :return:
    """
    order_keys = list(dict_items.keys())
    if how == 'len':
        ordered_weights = array([dict_items[k].shape[1] for k in order_keys])

    ordered_data = array([dict_items[k].T for k in order_keys])
    print('sizes of weights and data lists : {0} {1}'.format(len(ordered_weights), len(ordered_data)))
    b, lens_mod, pdfs_mod, inds = bin_packing_ffd_mod(ordered_weights, ordered_data, max_size, 0.01, ks_2samp_multi_dim)
    split_keys = [[order_keys[j] for j in ind_batch] for ind_batch in inds]
    return split_keys

