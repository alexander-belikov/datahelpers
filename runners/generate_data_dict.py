import argparse
import pandas as pd
from datahelpers.dftools import extract_idc_within_frequency_interval, count_elements_smaller_than_self_wdensity
import gzip
import pickle
from os.path import expanduser
from datahelpers.aux import str2bool
from numpy import ceil
from datahelpers.partition import partition_dict_to_subsamples
from bm_support.supervised import cluster_optimally_pd
from datahelpers.constants import ye, ai, ps, up, dn, ar, ni, nw, wi, cpop, cden


def main(df_type, version, present_columns,
         low_freq, hi_freq, low_bound_history_length, n_samples, test_head):

    with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(df_type, version)), 'rb') as fp:
        df = pickle.load(fp)

    ids = extract_idc_within_frequency_interval(df, ni, ps, (low_freq, hi_freq),
                                                low_bound_history_length)
    if test_head > 0:
        mask_ids = df[ni].isin(ids[:test_head])
        df = df.loc[mask_ids]
        ids = ids[:test_head]

    print('number of unique ids : {0}'.format(len(ids)))

    print(present_columns)

    if wi or nw in present_columns:
        df_ = df.groupby(ni).apply(lambda x: cluster_optimally_pd(x[ye], 2))
        extra_cols = set(df_.columns)
        df = pd.merge(df, df_, how='left', left_index=True, right_index=True)
        print(df.head())
        print('*** value counts of {0}'.format(wi))
        print(df[wi].value_counts())
        print('*** value counts of {0}'.format(nw))
        print(df.drop_duplicates(ni)[nw].value_counts())
        print(extra_cols)

    present_columns += list(set(extra_cols) - set(present_columns))

    if cpop in present_columns:
        df_ = df.groupby(ni).apply(lambda x: count_elements_smaller_than_self_wdensity(x[ye]))
        df_ = df_.rename(columns={0: cpop, 1: cden})
        df = pd.merge(df, df_, how='left', left_index=True, right_index=True)

    mask_ids = df[ni].isin(ids)
    up_dn = df.loc[mask_ids, [ni, up, dn]].drop_duplicates(ni).set_index(ni)
    means = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ps].mean()).rename('mean')
    sizes = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ps].shape[0]).rename('len')
    max_ye = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ye].max()).rename('max_year')
    min_ye = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ye].min()).rename('min_year')
    diff_ye = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ye].max() - x[ye].min()).rename('delta_year')

    df_stats = pd.concat([up_dn, means, sizes, max_ye, min_ye, diff_ye], axis=1).reset_index()
    df_stats['pop_density'] = df_stats['len'] / df_stats['delta_year']

    df_stats = df_stats.set_index(ni)
    print(df_stats.head())
    if test_head < 0:
        df_stats.to_csv(expanduser('~/data/kl/claims/pairs_freq_{0}_v_{1}_n_{2}'
                                   '_a_{3}_b_{4}.csv.gz'.format(df_type, version,
                                                                low_bound_history_length, low_freq, hi_freq)),
                        compression='gzip', index=True)

    data_dict = {}
    notify_fraction = 0.01
    n_notify = int(ceil(notify_fraction * len(ids)))
    print('n_notify: {0}'.format(n_notify))

    for idc in ids:
        data_dict[str(idc)] = df.loc[df[ni].isin([idc]), present_columns].values.T
        if len(data_dict) % n_notify == 0:
            print('{0:.3f}% fraction processed. {1}'.format(100 * len(data_dict) / len(ids),
                                                            len(data_dict)))

    total_size = sum(map(lambda x: x.shape[1], data_dict.values()))
    print('total size: {0}'.format(total_size))
    partition_dict = {k: data_dict[k].T for k in data_dict.keys()}

    idc_partition = partition_dict_to_subsamples(partition_dict, n_samples)

    print('number of batches in partition {0}; number of subsamples {1}'.format(idc_partition, n_samples))
    data_batches = [{k: data_dict[k] for k in sub} for sub in idc_partition]
    print('the actual number of subsamples {0}'.format(len(data_batches)))

    lens_ = [[sub_dict[k].shape[1] for k in sub_dict.keys()] for sub_dict in data_batches]
    print('lens: :{0}'.format(sorted(list(map(len, lens_)))))
    print('sums: :{0}'.format(sorted(list(map(sum, lens_)))))
    print('products {0}'.format(sorted(list(map(lambda x: sum(x)*len(x), lens_)))))
    print('products(+1) {0}'.format(sorted(list(map(lambda x: sum(x)*(len(x)+1), lens_)))))

    datatype = '_'.join(present_columns)

    if test_head < 0:
        with gzip.open(expanduser('~/data/kl/batches/data_batches_{0}_v_{1}_c_{2}_m_{3}_'
                                  'n_{4}_a_{5}_b_{6}.pgz'.format(df_type, version, datatype, n_samples,
                                                                 low_bound_history_length,
                                                                 low_freq, hi_freq)), 'wb') as fp:
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

    parser.add_argument('--data-columns', nargs='*', default=['year', 'ai', 'pos'])

    parser.add_argument('--test', type=int,
                        default=-1,
                        help='test on the head of the dataset')

    args = parser.parse_args()
    print(args._get_kwargs())

    low_f, hi_f = args.partition_sequence
    min_size_history = args.minsize_sequence

    main(args.datasource, args.version, args.data_columns,
         low_f, hi_f, min_size_history, args.nsamples, args.test)
