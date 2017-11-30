import argparse
import pandas as pd
import numpy as np
from datahelpers.dftools import extract_idc_within_frequency_interval
from sklearn.preprocessing import MinMaxScaler
import gzip
import pickle
from os.path import expanduser
from datahelpers.partition import partition_dict_to_subsamples

ye = 'year'
up = 'up'
dn = 'dn'
ni = 'new_index'
iden = 'identity'
idt = ni
ps = 'pos'
ai = 'ai'
ar = 'ar'


def main(df_type, version, present_columns, transform_columns,
         low_freq, hi_freq, low_bound_history_length, n_samples):

    with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(df_type, version)), 'rb') as fp:
        df = pickle.load(fp)

    # if ai in present_columns:
    #     alpha = 0.9
    #     mask = (df[ai] > alpha)
    #     df[ai + 'hi'] = 0
    #     df.loc[mask, ai + 'hi'] = 1
    #     df[ai] = (df[ai] - alpha) / (1 - alpha)
    #     print(sum(mask), sum(~mask))
    #     df.loc[~mask, ai] = 0
    #     present_columns.insert(present_columns.index(ai) + 1, ai + 'hi')

    print(present_columns)

    if ai in present_columns:
        ind_a = present_columns.index('ai')
        ind_b = present_columns.index('ai') + 1
    else:
        ind_b = len(present_columns)
        ind_a = ind_b - 1

    if iden in present_columns:
        df[iden] = 1

    dft = df[[ni] + present_columns].copy()

    ids = extract_idc_within_frequency_interval(dft, ni, ps, (low_freq, hi_freq),
                                                low_bound_history_length)
    print('number of unique ids : {0}'.format(len(ids)))
    print('feature indices : {0} {1}'.format(ind_a, ind_b))

    mask_ids = df[ni].isin(ids)
    up_dn = df.loc[mask_ids, [ni, up, dn]].drop_duplicates(ni).set_index(ni)
    means = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ps].mean()).rename('mean')
    sizes = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ps].shape[0]).rename('len')
    max_ye = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ye].max()).rename('max_year')
    min_ye = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ye].min()).rename('min_year')
    diff_ye = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ye].max() - x[ye].min()).rename('delta_year')

    df_stats = pd.concat([up_dn, means, sizes, max_ye, min_ye, diff_ye], axis=1).reset_index()
    # df_stats = df_stats.reindex(columns=df_stats.columns[[1, 2, 0, 3, 4, 5, 6, 7]])
    df_stats = df_stats.set_index(ni)
    print(df_stats.head())
    df_stats.to_csv(expanduser('~/data/kl/claims/pairs_freq_{0}_v_{1}_n_{2}'
                               '_a_{3}_b_{4}.csv.gz'.format(df_type, version,
                                                            low_bound_history_length, low_freq, hi_freq)),
                    compression='gzip', index=True)

    data_dict = {}
    # size = len
    notify_fraction = 0.01
    n_notify = int(notify_fraction*len(ids))
    print('n_notify: {0}'.format(n_notify))
    for idc in ids:
        data_dict[str(idc)] = dft.loc[dft[ni].isin([idc]), present_columns].values.T
        if len(data_dict) % n_notify == 0:
            print('{0:.3f} fraction processed. {1}'.format(notify_fraction*(len(data_dict)//n_notify),
                                                           len(data_dict)))

    data_dict2 = {}

    for c in transform_columns:
        if c in present_columns:
            index = present_columns.index(c)
            for k, d in data_dict.items():
                sc = MinMaxScaler()
                d2 = d.copy().astype(np.float)
                d2[index] = np.squeeze(sc.fit_transform(d[index].reshape(-1, 1)))
                data_dict2[k] = d2

    total_size = sum(map(lambda x: x.shape[1], data_dict2.values()))
    print('total size: {0}'.format(total_size))
    partition_dict = {k: data_dict2[k][ind_a:ind_b].T for k in data_dict2.keys()}

    idc_partition = partition_dict_to_subsamples(partition_dict, n_samples)

    print('number of batches in partition {0}; number of subsamples {1}'.format(idc_partition, n_samples))
    data_batches = [{k: data_dict2[k] for k in sub} for sub in idc_partition]
    print('the actual number of subsamples {0}'.format(len(data_batches)))

    lens_ = [[sub_dict[k].shape[1] for k in sub_dict.keys()] for sub_dict in data_batches]
    print('lens: :{0}'.format(sorted(list(map(len, lens_)))))
    print('sums: :{0}'.format(sorted(list(map(sum, lens_)))))
    print('products {0}'.format(sorted(list(map(lambda x: sum(x)*len(x), lens_)))))
    print('products(+1) {0}'.format(sorted(list(map(lambda x: sum(x)*(len(x)+1), lens_)))))

    datatype = '_'.join(present_columns)

    with gzip.open(expanduser('~/data/kl/batches/data_batches_{0}_v_{1}_c_{2}_m_{3}_'
                              'n_{4}_a_{5}_b_{6}.pgz'.format(df_type, version, datatype, n_samples,
                                                             low_bound_history_length, low_freq, hi_freq)), 'wb') as fp:
        pickle.dump(data_batches, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # gw

    # version = 1 for lit

    parser.add_argument('-d', '--datasource',
                        default='gw',
                        help='type of data to work with [gw, lit]')

    parser.add_argument('-v', '--version',
                        default=8, type=int,
                        help='version of data source')

    parser.add_argument('-s', '--nsamples',
                        default=8, type=int,
                        help='size of data batches')

    parser.add_argument('-n', '--minsize-sequence',
                        default=20, type=int,
                        help='version of data source')

    parser.add_argument('-p', '--partition-sequence',
                        nargs='+', default=[0.1, 0.9], type=float,
                        help='define interval of observed freqs for sequence consideration')

    parser.add_argument('--data-columns', nargs='*', default=['year', 'identity', 'ai', 'pos'])

    parser.add_argument('--transform-columns', nargs='*', default=['year'])

    args = parser.parse_args()
    print(args._get_kwargs())

    low_f, hi_f = args.partition_sequence
    min_size_history = args.minsize_sequence

    main(args.datasource, args.version, args.data_columns, args.transform_columns,
         low_f, hi_f, min_size_history, args.nsamples)
