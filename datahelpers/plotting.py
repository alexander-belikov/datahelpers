import matplotlib.pyplot as plt
import numpy as np
from constants import ye


def plot_hist_true_false(dft, fname=None):
    """

    :param dft:
    :param fname:
    :return:
    """

    data = dft[ye]
    min_data = min(data)
    max_data = max(data)
    m = (dft['negative'] == False)
    data_pos = dft.loc[m, ye]
    data_neg = dft.loc[~m, ye]

    fig = plt.figure(figsize=(6, 6))
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)

    N = 10
    delta_x = max(int((max_data - min_data)/N), 1)
    x_bins = np.arange(min_data, max_data + 2*delta_x, delta_x)
    x_ticks = np.arange(min_data-delta_x, max_data + 3*delta_x, delta_x)
    x_labels = [str(int(t)) for t in x_ticks]

    xranges = [min_data, max_data]
    yranges = [0, 1e1]
    plt.xlim(xranges)
    plt.ylim(yranges)

    opacity = 0.3
    plt.xticks(x_ticks, x_labels)
    ll = plt.hist([data_pos+1e-6, data_neg+1e-6], bins=x_bins, color=['b', 'r'],
                  alpha=opacity, stacked=True, rwidth=delta_x)
    if fname:
        plt.savefig(fname)


def plot_hist(arr_list, ymax=None, fname=None):
    """

    :param arr_list:
    :param ymax:
    :param fname:
    :return:
    """
    fig = plt.figure(figsize=(6,6))
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)

    mins = [min(d) for d in arr_list]
    maxs = [max(d) for d in arr_list]
    min_data = min(mins)
    max_data = max(maxs)

    N = 10

    delta_x = max(int((max_data - min_data)/N), 1)
    print min_data, max_data, delta_x
    x_bins = np.arange(min_data, max_data + 2*delta_x, delta_x)
    x_ticks = np.arange(min_data-delta_x, max_data + 3*delta_x, delta_x)
    x_labels = [str(int(t)) for t in x_ticks]

    xranges = [min_data, max_data]
    plt.xlim(xranges)
    if not ymax:
        ymax = max([d.shape[0] for d in arr_list])
    plt.ylim([0, ymax])

    opacity = 0.3
    plt.xticks(x_ticks, x_labels)
    for arr in arr_list:
        ll = plt.hist(arr, bins=x_bins,
                      alpha=opacity, rwidth=delta_x)
    if fname:
        plt.savefig(fname)
