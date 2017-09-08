from numpy import zeros, arange, array, concatenate, \
    argsort, abs, sum, argmin, std, mean, tile, argwhere, where, ceil
from scipy.stats import ks_2samp
from copy import deepcopy

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
                # print(sample0.shape, ks_cur, ks_cur2)
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
    # p_val is not additive
    s = 0
    for x, y in zip(sample_a.T, sample_b.T):
        r = ks_2samp(x, y)
        s += r[0]
    p_val = r[1]
    return s, p_val


def partition_dict(dict_items, ind_start, ind_end, max_size, how='len'):
    """

    :param dict_items: a dict of numpy arrays with equal first dimension
    :param ind_start: index of first ar
    :param ind_end:
    :param max_size:
    :return:
    """
    order_keys = list(dict_items.keys())
    if how == 'len':
        ordered_weights = array([dict_items[k].shape[1] for k in order_keys])

    ordered_data = array([dict_items[k][ind_start:ind_end].T for k in order_keys])
    print('sizes of weights and data lists : {0} {1}'.format(len(ordered_weights), len(ordered_data)))
    b, lens_mod, pdfs_mod, inds = bin_packing_ffd_mod(ordered_weights, ordered_data, max_size, 0.01, ks_2samp_multi_dim)
    split_keys = [[order_keys[j] for j in ind_batch] for ind_batch in inds]
    return split_keys


def try_swapping_elements(item_a, item_b, mean_phi_over_weights, sample0,
                          epsilon=0.5, distance_func=ks_2samp):
    # swap ith and jth elements from pdf_a and pdf_b and w_a and w_b

    # epsilon controls how much importance is given to weight based metric vs pdf metric
    # epsilon = 1 only pdf metric; epsilon = 0 only weight based metric
    weight_a, pdf_a = item_a
    weight_b, pdf_b = item_b
    if not (len(weight_a) == len(pdf_a) and len(weight_b) == len(pdf_b)):
        raise ValueError('cardinatlity of indices, weights and pdfs are not equal')
    len_a, len_b = len(weight_a), len(weight_b)
    sum_a, sum_b = sum(weight_a), sum(weight_b)
    pi_a, pi_b = len_a * sum_a, len_b * sum_b
    # how far pa and pb from the mean
    # delta_a, delta_b = pi_a - mean_phi_over_weights, pi_b - mean_phi_over_weights
    da0 = distance_func(concatenate(pdf_a), sample0)[0]
    db0 = distance_func(concatenate(pdf_b), sample0)[0]

    # a_ij = w^a_i - w^b_j
    delta_matrix = tile(weight_a, (len(weight_b), 1)).T - array(weight_b)
    diff_a = abs(pi_a - mean_phi_over_weights) - abs(pi_a - len_a*delta_matrix - mean_phi_over_weights)
    diff_b = abs(pi_b - mean_phi_over_weights) - abs(pi_b + len_b*delta_matrix - mean_phi_over_weights)
    pairs = argwhere((diff_a > 0) & (diff_b > 0))

    pairs_metric = [(diff_a[x, y] + diff_b[x, y]) for x, y in pairs]
    pairs_metric_sorted_inds = argsort(pairs_metric)
    pairs_sorted = [pairs[j] for j in pairs_metric_sorted_inds]
    swap_flag = False
    ja, jb = -1, -1
    # make sure first index is a lower number of items in the bin
    while pairs_sorted and not swap_flag:
        ja, jb = pairs_sorted.pop()
        pdf_a_, pdf_b_ = deepcopy(pdf_a), deepcopy(pdf_b)
        pdf_aj, pdf_bj = pdf_a_.pop(ja), pdf_b_.pop(jb)
        pdf_a_.append(pdf_bj)
        pdf_b_.append(pdf_aj)
        da = distance_func(concatenate(pdf_a_), sample0)[0]
        db = distance_func(concatenate(pdf_b_), sample0)[0]

        # pi_a_prime = (sum_a - weight_a[ia] + weight_b[ib])*len_a
        # pi_b_prime = (sum_b - weight_b[ib] + weight_a[ia])*len_b
        # print(pi_b_prime - pi_a_prime, diff_a[ia, ib] + diff_b[ia, ib], weight_a[ia], weight_b[ib])
        # print(da-da0, db-db0)
        if da < da0 and db < db0:
            swap_flag = True
    return swap_flag, (ja, ja)


def try_moving_element(item_a, item_b, mean_phi_over_weights, sample0,
                       epsilon=0.5, distance_func=ks_2samp):
    # given
    # weights_a, pdf_a = item_a 
    # weights_b, pdf_b = item_b
    # take one kth element from weights_b, pdf_b in such a way that
    # la'*sa' is closer mean_phi_over_weights and 
    # distances rho(pdf_a', sample0) and rho(pdf_b', sample0) are improved

    # epsilon controls how much importance is given to weight based metric vs pdf metric
    # epsilon = 1 only pdf metric; epsilon = 0 only weight based metric
    weight_a, pdf_a = item_a
    weight_b, pdf_b = item_b
    if not (len(weight_a) == len(pdf_a) and len(weight_b) == len(pdf_b)):
        raise ValueError('cardinatlity of indices, weights and pdfs are not equal')
    len_a, len_b = len(weight_a), len(weight_b)
    sum_a, sum_b = sum(weight_a), sum(weight_b)
    delta_a, delta_b = abs(len_a * sum_a - mean_phi_over_weights), abs(len_b * sum_b - mean_phi_over_weights)
    da0 = distance_func(concatenate(pdf_a), sample0)[0]
    db0 = distance_func(concatenate(pdf_b), sample0)[0]
    pdf_dist = []
    pi_dist = []
    for j in range(len(weight_b)):
        # da < da0 - good(!)
        # db < db0 - good(!)
        da = distance_func(concatenate([pdf_b[j]] + pdf_a), sample0)[0] - da0
        db = distance_func(concatenate([pdf_b[k] for k in range(len_b) if k != j]), sample0)[0] - db0

        pa = abs(sum([weight_b[j]] + weight_a) * (len_a + 1) - mean_phi_over_weights)
        pb = abs(sum([weight_b[k] for k in range(len_b) if k != j]) * (len_b - 1) - mean_phi_over_weights)
        pi_dist.append(pa + pb)
        pdf_dist.append((da, db))

    pi_dist_arr = array(pi_dist) / (delta_a + delta_b)
    pdf_dist_arr = array(pdf_dist)
    pdf_dist_arr /= abs(pdf_dist_arr.max(axis=0))
    pdf_dist_arr = abs(pdf_dist_arr)

    # convolved pdf distances
    pdf_dist_conv = (pdf_dist_arr ** 2).sum(axis=1)
    distances_normed = (epsilon * pdf_dist_conv + (1. - epsilon) * pi_dist_arr ** 2) ** 0.5
    i_best = argmin(distances_normed)
    if pi_dist_arr[i_best] < 1.0 and pdf_dist_conv[i_best] < 1.0:
        move_flag = True
    else:
        move_flag = False
    return move_flag, (None, i_best)


def manage_lists(partition_inds, weights, pdfs, sample0, mask_func,
                 foo=try_moving_element, distance_func=ks_2samp_multi_dim):

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
        accepted, (j_a, j_b) = foo((bin_a, pdf_a), (bin_b, pdf_b), mean_ps, sample0, 0.5, distance_func)
        if accepted:
            partition_ind_a, partition_ind_b = list(partition_inds[index_a]), list(partition_inds[index_b])
            if j_a:
                partition_ind_a.pop(j_a)
            if j_b:
                partition_ind_b.pop(j_b)
            if j_a:
                partition_ind_b += [partition_inds[index_a][j_a]]
            if j_b:
                partition_ind_a += [partition_inds[index_b][j_b]]

            pps = [pp for pp in pps if pp[0] != ia and pp[1] != ib]
            # print(partition_ind_a, partition_inds[index_a])
            # print(partition_ind_b, partition_inds[index_b])
            partition_inds[index_a] = partition_ind_a
            partition_inds[index_b] = partition_ind_b
    return partition_inds


def reshuffle_bins(partition_indices, weights, pdfs, distance_func=ks_2samp):
    partition_indices_new = deepcopy(partition_indices)
    sample0 = concatenate(pdfs)
    partition_indices_new = manage_lists(partition_indices_new, weights, pdfs, sample0,
                                         lambda x: x > 1, try_moving_element, distance_func)

    print(check_packing(partition_indices_new, weights, pdfs))
    partition_indices_new = manage_lists(partition_indices_new, weights, pdfs, sample0,
                                         lambda x: x == 1, try_moving_element, distance_func)
    print(check_packing(partition_indices_new, weights, pdfs))

    partition_indices_new = manage_lists(partition_indices_new, weights, pdfs, sample0,
                                         lambda x: x == 1, try_swapping_elements, distance_func)
    print(check_packing(partition_indices_new, weights, pdfs))

    # ls = list(map(len, bins))
    # ss = list(map(sum, bins))
    # ps = list(map(lambda x: x[0] * x[1], zip(ls, ss)))

    # ps_sorted_inds = argsort(ps)
    # ls_sorted = array(ls)[ps_sorted_inds]
    # ps_sorted = array(ps)[ps_sorted_inds]
    # ls_matrix = tile(ls_sorted, (len(ls), 1)).T - ls_sorted
    # pairs = argwhere(ls_matrix == 1)[:, ::-1]
    # dd = {k: list(pairs[where(pairs[:, 0] == k)][::-1, 1])
    #       for k in list(set(pairs[:, 0]))}
    #
    # for k in dd:
    #     print(ps_sorted[k], ps_sorted[dd[k]])
    #     print(ls_sorted[k], ls_sorted[dd[k]])

    return partition_indices_new


def bin_packing_mean(weights, pdfs, n, min_batch=3, rescue_size=1, distance_func=ks_2samp,
                     violation_level_mean=0.1, violation_level_pdf=0.01):
    w_mean0 = mean(weights)
    sample0 = concatenate(pdfs)
    items_per_bin = int(ceil(len(weights) / n))
    bin_capacity = (items_per_bin * w_mean0)
    bin_product = bin_capacity * items_per_bin

    # descending order
    inds_sorted = argsort(weights)[::-1]
    inds2 = list(inds_sorted)
    weights2 = list(weights[inds_sorted])
    pdfs2 = list(pdfs[inds_sorted])

    if max(weights) > bin_capacity:
        raise ValueError('Max item weight is greater than proposed bin cap')
    # populate each bin with a largest available element
    bins = [[x] for x in weights2[:n]]
    indices = [[i] for i in inds2[:n]]
    pdf_bins = [[i] for i in pdfs2[:n]]

    bins_output = []
    indices_output = []
    pdfs_output = []
    weights2 = weights2[n:]
    inds2 = inds2[n:]
    pdfs2 = pdfs2[n:]
    ind_cur_bin = 0
    loop_counter = 0

    while weights2:
        add_items_cnt = min([items_per_bin - len(bins[ind_cur_bin]), len(weights2), min_batch])
        accepted = False
        cur_bin = bins[ind_cur_bin]
        cur_ind_bin = indices[ind_cur_bin]
        cur_pdf_bin = pdf_bins[ind_cur_bin]
        while not accepted and add_items_cnt > 0:
            cur_bin_ = list(cur_bin)
            cur_bin_.extend(weights2[-add_items_cnt:])
            w_mean = mean(cur_bin)
            w_proposed = mean(cur_bin_)
            decision = abs(w_mean - w_mean0) - abs(w_proposed - w_mean0)
            if decision > -violation_level_mean * std(cur_bin_) and sum(cur_bin_) * len(cur_bin_) < bin_product:
                ks_cur = distance_func(concatenate(cur_pdf_bin), sample0)
                ks_cur2 = distance_func(concatenate(cur_pdf_bin + pdfs2[-add_items_cnt:]), sample0)
                improves_pdf = (ks_cur2[0] < ks_cur[0] + violation_level_pdf)
                if improves_pdf:
                    cur_ind_bin.extend(inds2[-add_items_cnt:])
                    cur_pdf_bin.extend(pdfs2[-add_items_cnt:])
                    weights2 = weights2[:-add_items_cnt]
                    inds2 = inds2[:-add_items_cnt]
                    pdfs2 = pdfs2[:-add_items_cnt]
                    bins[ind_cur_bin] = cur_bin_
                    accepted = True
                    if len(cur_bin_) >= items_per_bin:
                        bins_output.append(bins.pop(ind_cur_bin))
                        indices_output.append(indices.pop(ind_cur_bin))
                        pdfs_output.append(pdf_bins.pop(ind_cur_bin))
            if not accepted:
                add_items_cnt -= 1
        if accepted:
            loop_counter = 0
        else:
            loop_counter += 1
        if loop_counter > len(bins):
            # rescue plan : place weights[-1] into a bin with least damage
            candidate = weights2[-rescue_size:]
            mean_distance = array(list(map(lambda x: abs(mean(x + candidate) - w_mean0), bins)))
            mean_distance = mean_distance / max(mean_distance)
            pdf_distance = array(list(map(lambda x:
                                          distance_func(concatenate(x + pdfs2[-rescue_size:]), sample0)[0], pdf_bins)))
            pdf_distance = pdf_distance / max(pdf_distance)
            dist = (pdf_distance ** 2 + mean_distance ** 2) ** 0.5
            j = argmin(dist)
            bins[j].extend(candidate)
            indices[j].extend(inds2[-rescue_size:])
            pdf_bins[j].extend(pdfs2[-rescue_size:])
            weights2 = weights2[:-rescue_size]
            inds2 = inds2[:-rescue_size]
            pdfs2 = pdfs2[:-rescue_size]
        if bins:
            ind_cur_bin = (ind_cur_bin + 1) % len(bins)
    bins_output.extend(bins)
    indices_output.extend(indices)
    pdfs_output.extend(pdfs)
    return indices_output


def check_packing(list_indices, weights, pdfs, distance_func=ks_2samp_multi_dim):
    if not (sum(list(map(len, list_indices))) == len(weights) and len(weights) == len(pdfs)):
        raise ValueError('cardinatlity of indices, weights and pdfs are not equal')

    sample0 = concatenate(pdfs)

    pdf_bins = [[pdfs[j] for j in ind_batch] for ind_batch in list_indices]
    bins = [[weights[j] for j in ind_batch] for ind_batch in list_indices]
    ls = list(map(len, bins))
    ss = list(map(sum, bins))
    ps = list(map(lambda x: x[0] * x[1], zip(ls, ss)))
    mean_ps = mean(ps)
    std_ps = std(ps)
    rhos = list(map(lambda x: distance_func(concatenate(x), sample0)[0], pdf_bins))
    return mean_ps, std_ps, min(rhos), max(rhos)
