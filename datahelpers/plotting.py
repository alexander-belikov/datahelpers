import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .constants import ye
from bm_support.math_aux import find_intlike_delta
from matplotlib.pyplot import subplots
from seaborn import set_style
from seaborn import plt as sns_plt
from os import mkdir
from os.path import exists
from functools import partial
from bm_support.math_aux import np_logistic_step
from numpy import array, arange, mean, floor, ceil


def plot_hist_true_false(dft, N=5, fname=None, x_column_name=ye,
                         y_column_name='negative', normed_flag=False,
                         title='', alpha=0.7,
                         sns_style='darkgrid', integer_bbs=True,
                         squeeze_uniform=False):

    data = dft[x_column_name]
    min_data = min(data)
    max_data = max(data)
    if integer_bbs:
        min_data = floor(min_data)
        max_data = ceil(max_data)

    m = (dft[y_column_name] == 1)
    data_pos = dft.loc[m, x_column_name]
    data_neg = dft.loc[~m, x_column_name]

    fig = plt.figure(figsize=(6, 6))
    rect = [0.15, 0.15, 0.75, 0.75]
    ax = fig.add_axes(rect)
    ax.set_title(title)

    delta_x = max(int((max_data - min_data)/N), 1)
    x_bins = np.arange(min_data, max_data + 2*delta_x, delta_x)
    x_centers = x_bins[1:] - 0.5*delta_x

    x_ticks = np.arange(min_data-delta_x, max_data + 3*delta_x, delta_x)
    x_labels = [str(int(t)) for t in x_ticks]

    xranges = [min_data, max_data]
    plt.xlim(xranges)

    plt.xticks(x_ticks, x_labels)
    sns.set_style(sns_style)

    # hist_kw = {
    #             # 'histtype': 'step',
    #             # 'histtype': 'stepfilled',
    #             # 'bins': x_bins,
    #             'alpha': opacity,
    #             'rwidth': delta_x,
    #             'stacked': True,
    #             'normed': normed_flag,
    #             # 'color': ['b', 'r'],
    #             'lw': linewidth}
    #
    # ldata = [data_pos + 1e-6, data_neg + 1e-6]
    # lcolors = ['b', 'r']
    # for arr, c in zip(ldata, lcolors):
    #     hist_kw['color'] = c
    #     ll = sns.distplot(arr, bins=x_bins, hist_kws=hist_kw, kde=False)

    if squeeze_uniform:
        p_binned, _ = np.histogram(data_pos, x_bins)
        n_binned, _ = np.histogram(data_neg, x_bins)
        s_binned = p_binned+n_binned
        p_binned = p_binned/s_binned
        n_binned = n_binned/s_binned

        plt.bar(x_centers, p_binned, width=delta_x, alpha=alpha)
        plt.bar(x_centers, n_binned, bottom=p_binned, width=delta_x, alpha=alpha)

    else:
        ll = plt.hist([data_pos+1e-6, data_neg+1e-6], bins=x_bins,
                      alpha=alpha, stacked=True, rwidth=delta_x, normed=normed_flag)
    if fname:
        plt.savefig(fname)
    return ax


def plot_hist_float_x(data, xranges, yranges=None, N=10, opacity=0.8, ylog_axis=False, fname=None):
    """

    :param data:
    :param xranges:
    :param N:
    :param opacity:
    :param ylog_axis:
    :param fname:
    :return:
    """

    fig = plt.figure(figsize=(6, 6))
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    if ylog_axis:
        ax.set_yscale('log')
    plt.xlim(xranges)
    if yranges:
        plt.ylim(yranges)

    ll = plt.hist(data, bins=np.linspace(xranges[0], xranges[1], N+1), alpha=opacity)
    if fname:
        plt.savefig(fname)
    return ll


def plot_hist(arr_list, approx_nbins=10, ylog_axis=False,
              xticks_factor=1, normed_flag=False, opacity=0.5, linewidth=2,
              fname=None, y_axis_mult=None, title='', integerize=False,
              xlabels_style='int', sns_style='darkgrid'):
    """

    :param arr_list:
    :param approx_nbins:
    :param ylog_axis:
    :param xticks_factor:
    :param normed_flag:
    :param opacity:
    :param linewidth:
    :param fname:
    :param y_axis_mult:
    :param title:
    :param integerize:
    :param xlabels_style: 'int', 'sci' or 'float'
    :param sns_style:
    :return:
    """
    fig = plt.figure(figsize=(6, 6))
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_title(title)

    mins = [min(d) for d in arr_list]
    maxs = [max(d) for d in arr_list]
    if integerize:
        min_data = floor(min(mins))
        max_data = ceil(max(maxs))
    else:
        min_data = min(mins)
        max_data = max(maxs)

    print(min_data, max_data)
    delta_x = find_intlike_delta(min_data, max_data, approx_nbins)
    print(min_data, max_data, delta_x)
    x_bins = np.arange(min_data, max_data + 2*delta_x, delta_x)
    print(x_bins)
    x_ticks = np.arange(min_data, max_data + 3*delta_x, delta_x)
    x_ticks = x_ticks[::xticks_factor]
    if xlabels_style == 'int':
        x_labels = [str(int(t)) for t in x_ticks]
    elif xlabels_style == 'sci':
        x_labels = ['{0:.1e}'.format(t) for t in x_ticks]
    elif xlabels_style == 'float':
        x_labels = ['{0:.1f}'.format(t) for t in x_ticks]

    xrange = [min_data, max_data]
    plt.xlim(xrange)

    # if not normed_flag:
    #     if not yrange:
    #         yrange = [1e0, max([d.shape[0] for d in arr_list])]
    #     plt.ylim(yrange)

    if ylog_axis:
        ax.set_yscale('log')

    sns.set_style(sns_style)
    # sns.set_palette('bright')

    plt.xticks(x_ticks, x_labels)
    hist_kw = {
                'histtype': 'step',
                # 'histtype': 'stepfilled',
                'alpha': opacity,
                'rwidth': delta_x,
                'normed': normed_flag,
                'lw': linewidth}
    for arr in arr_list:
        tmp_arr = min(arr) + (1. - 1e-8)*(arr - min(arr))

        ll = sns.distplot(tmp_arr, bins=x_bins, hist_kws=hist_kw, kde=False)

    if y_axis_mult:
        yr = ax.get_ylim()
        yr_new = list(yr)
        yr_new[1] = yr[1] + round(y_axis_mult*yr[1], -1)
        ax.set_ylim(yr_new)

    if fname:
        plt.savefig(fname)


def plot_beta_steps(n_features_ext, dict_base, raw_best_dict_plot_beta, tlow, thi, sorted_first,
                    fname_prefix=None, path='./', format='pdf'):

    xr = floor(tlow), ceil(thi)
    xs = arange(xr[0], xr[1], 0.1)

    for j in range(n_features_ext):
        dict_rename = {k + str(j): dict_base[k] for k in dict_base.keys()}
        align_list = ['b1', 'b2', 't0', 'g']
        pps_der = {dict_rename[k]: raw_best_dict_plot_beta[k] for k in dict_rename.keys()}
        # pprint(pps_der)
        pps_der_list = [pps_der[k] for k in align_list]

        llists = [pps_der_list]
        foos = [partial(np_logistic_step, *ps) for ps in llists]
        yss = [map(foo, xs) for foo in foos]
        f, ax = subplots(figsize=(7, 7))

        ax.set_xlim(xr)
        set_style("darkgrid")

        lss = ['-', '--']
        for ys, ls in zip(yss, lss):
            sns_plt.plot(xs, ys, ls)

        if fname_prefix:
            if not exists(path):
                mkdir(path)
            if not path.endswith('/'):
                path += '/'
            plt.savefig("%s%s_%s%s.%s" % (path, fname_prefix, 'bestfit_feature_', str(j), format))

    for j in range(n_features_ext):
        f, ax = subplots(figsize=(7, 7))
        xr = floor(tlow), ceil(thi)
        ax.set_xlim(xr)
        #     ax.set_ylim([beta_min, beta_max])
        xs = arange(xr[0], xr[1], 0.1)
        set_style("darkgrid")
        dict_base = {'betaCenter_': 't0', 'betaLeft_': 'b1', 'betaRight_': 'b2', 'betaSteep_': 'g'}
        dict_rename = {k + str(j): dict_base[k] for k in dict_base.keys()}
        align_list = ['b1', 'b2', 't0', 'g']
        #     pps_plot_beta = {k : pps[k] for k in pps if 'beta' in k and not 'xprior' in k}
        #     pps_orig = {}
        #     pps_orig = {dict_rename[k]:pps_plot_beta[k] for k in dict_rename.keys()}
        #     pps_orig_list = [pps_orig[k] for k in align_list]
        #     ys = map(partial(np_logistic_step, *pps_orig_list), xs)
        #     sns.plt.plot(xs, ys, ls='-', lw=2)

        llist = []

        for rbdp in sorted_first:
            raw_best_dict_plot_beta = {k: rbdp[1][k]
                                       for k in sorted_first[0][1].keys()
                                       if 'beta' in k and not 'xprior' in k}
            pps_der = {dict_rename[k]: raw_best_dict_plot_beta[k] for k in dict_rename.keys()}
            pps_der_list = [pps_der[k] for k in align_list]

            llist.append(pps_der_list)

        foos = [partial(np_logistic_step, *ps) for ps in llist]
        yss = [map(foo, xs) for foo in foos]

        means = mean(array(llist), axis=0)
        ys = map(partial(np_logistic_step, *means), xs)
        sns_plt.plot(xs, ys, ls='-', lw=2)

        for ys in yss:
            sns_plt.plot(xs, ys, ls='--', lw=0.5)

        if fname_prefix:
            if not exists(path):
                mkdir(path)
            if not path.endswith('/'):
                path += '/'
            plt.savefig("%s%s_%s%s.%s" % (path, fname_prefix, 'fits_stats_feature_', str(j), format))

