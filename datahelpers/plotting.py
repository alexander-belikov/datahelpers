import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from constants import ye
from bm_support.math_aux import find_intlike_delta


def plot_hist_true_false(dft, N=5, fname=None, x_column_name=ye,
                         y_column_name='negative', normed_flag=False,
                         linewidth=1.0, title='',
                         sns_style='darkgrid'):

    data = dft[x_column_name]
    min_data = min(data)
    max_data = max(data)
    m = (dft[y_column_name] == 1)
    data_pos = dft.loc[m, x_column_name]
    data_neg = dft.loc[~m, x_column_name]

    fig = plt.figure(figsize=(6, 6))
    rect = [0.15, 0.15, 0.75, 0.75]
    ax = fig.add_axes(rect)
    ax.set_title(title)

    delta_x = max(int((max_data - min_data)/N), 1)
    x_bins = np.arange(min_data, max_data + 2*delta_x, delta_x)
    x_ticks = np.arange(min_data-delta_x, max_data + 3*delta_x, delta_x)
    x_labels = [str(int(t)) for t in x_ticks]

    xranges = [min_data, max_data]
    plt.xlim(xranges)

    opacity = 0.3
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
    sns.set_style("darkgrid")
    ll = plt.hist([data_pos+1e-6, data_neg+1e-6], bins=x_bins, color=['b', 'r'],
                  alpha=opacity, stacked=True, rwidth=delta_x, normed=normed_flag)
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


def plot_hist(arr_list, approx_nbins=10, yrange=[], ylog_axis=False,
              xticks_factor=1, normed_flag=False, opacity=0.5, linewidth=2,
              fname=None, y_axis_mult=None, title='', integerize=False,
              int_xlabels=True,
              sns_style='darkgrid'):
    """

    :param arr_list:
    :param approx_nbins:
    :param yrange:
    :param ylog_axis:
    :param xticks_factor:
    :param normed_flag:
    :param opacity:
    :param fname:
    :return:
    """
    fig = plt.figure(figsize=(6, 6))
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_title(title)

    mins = [min(d) for d in arr_list]
    maxs = [max(d) for d in arr_list]
    if integerize:
        min_data = int(min(mins))
        max_data = int(max(maxs))
    else:
        min_data = min(mins)
        max_data = max(maxs)

    print min_data, max_data
    delta_x = find_intlike_delta(min_data, max_data, approx_nbins)
    x_bins = np.arange(min_data, max_data + 2*delta_x, delta_x)
    x_ticks = np.arange(min_data, max_data + 3*delta_x, delta_x)
    x_ticks = x_ticks[::xticks_factor]
    if int_xlabels:
        x_labels = [str(int(t)) for t in x_ticks]
    else:
        x_labels = [str(t) for t in x_ticks]

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
        ll = sns.distplot(arr - 1e-6, bins=x_bins, hist_kws=hist_kw, kde=False)

    if y_axis_mult:
        yr = ax.get_ylim()
        yr_new = list(yr)
        yr_new[1] = yr[1] + round(y_axis_mult*yr[1], -1)
        ax.set_ylim(yr_new)

    if fname:
        plt.savefig(fname)
