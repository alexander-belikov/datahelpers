import argparse
import pandas as pd
from datahelpers.dftools import extract_idc_within_frequency_interval, count_elements_smaller_than_self_wdensity
from datahelpers.dftools import calculate_uniformity_ks
import gzip
import pickle
from os.path import expanduser
from numpy import ceil
from datahelpers.partition import partition_dict_to_subsamples
from bm_support.supervised import cluster_optimally_pd, optimal_2split_pd
from datahelpers.community_tools import produce_cluster_df, project_weight
from datahelpers.constants import ye, ai, ps, up, dn, ar, ni, nw, wi, cpop, pm, cden
from tqdm import tqdm
from hashlib import sha1


def main(df_type, version, feature_groups,
         low_freq, hi_freq, low_bound_history_length, n_samples, test_head):

    with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(df_type, version)), 'rb') as fp:
        df = pickle.load(fp)

    dft = df.copy()

    # feature_groups = ['mainfeatures', 'cycle', 'density', 'normeddensity']
    columns_dict = {}
    if 'mainfeatures' in feature_groups:
        columns_dict['mainfeatures'] = [ni, pm, ye, ai, ar]

    ids = extract_idc_within_frequency_interval(df, ni, ps, (low_freq, hi_freq),
                                                low_bound_history_length)
    if test_head > 0:
        ids = ids[:test_head]

    mask_ids = df[ni].isin(ids)
    df = df.loc[mask_ids]

    print('number of unique ids : {0}'.format(len(ids)))

    print('feature groups', feature_groups)

    tqdm.pandas(tqdm())

    windows = [None, 1, 2, 3]
    right_conts = [False, True]
    density_columns = []
    # rc stand for right continuous : in the popularity count
    # instead of `<` for the past statements, we use `<=`

    if 'density' in feature_groups:
        for w in windows:
            for rc_flag in right_conts:
                df_ = df.groupby(ni).progress_apply(lambda x: count_elements_smaller_than_self_wdensity(x[ye],
                                                                                                        w, rc_flag))
                df = pd.merge(df, df_, how='left', left_index=True, right_index=True)
                density_columns.extend(list(df_.columns))

    columns_dict['density'] = density_columns

    ks_columns = []
    if 'ksst' in feature_groups:
        for w in windows:
            for rc_flag in right_conts:
                df_ = df.groupby(ni).progress_apply(lambda x: calculate_uniformity_ks(x[ye], w, rc_flag))
                df = pd.merge(df, df_, how='left', left_index=True, right_index=True)
                ks_columns.extend(list(df_.columns))

    columns_dict['ksst'] = ks_columns

    # network clustering // community detection
    # detect communities on the full network
    if 'density' in feature_groups and 'normeddensity' in feature_groups:

        normed_density_columns = [c + '_normed' for c in density_columns]

        df_edges = project_weight(dft)
        df_cc, rep = produce_cluster_df(df_edges, cutoff_frac=0.0, unique_edges=False, cut_nodes=False, verbose=True)
        print('time on comm. detection {0:.3f}'.format(rep))
        print(sum(df_cc['domain_id_up'].isnull()), sum(df_cc['domain_id_dn'].isnull()))
        # print(df_cc.dtypes, df_cc.shape, sum(df_cc['domain_id_up'].isnull()))
        dup, ddn = 'domain_id_up', 'domain_id_dn'
        df_dd_cnt = df_cc.groupby([dup, ddn], group_keys=False).apply(lambda x: x['weight'].sum())

        # obtain domain_up, domain_dn with 'norm' being the sum of edges between domain types
        df_dd = df_dd_cnt.reset_index().rename(columns={0: 'norm'})

        # merge df_dd back to df_cc (up, dn ,dup, ddn, weight)
        df_cc2 = pd.merge(df_cc, df_dd, on=(dup, ddn), how='left')

        df2 = pd.merge(df, df_cc2, on=(up, dn), how='left')
        if sum(~(df2['norm'] > 0)) > 0:
            print('break : some norms are negative')

        for c, c2 in zip(density_columns, normed_density_columns):
            df2[c2] = df2[c] / df2['norm']

        diffuse_cluster_id = df_cc['domain_id_up'].max()

        mask = (df2[dup] == diffuse_cluster_id) | (df2[ddn] == diffuse_cluster_id)
        df2.loc[mask, 'norm'] = 0
        df2['diffuse'] = 0.0
        df2.loc[mask, 'diffuse'] = 1.0

        columns_dict['normeddensity'] = normed_density_columns + ['norm', 'diffuse']
        df = df2

    # time clustering
    if 'cycle' in feature_groups:
        print('starting optimal clustering:')
        # naive split
        df_ = df.groupby(ni).progress_apply(lambda x: optimal_2split_pd(x[ye]))
        # kmeans split
        # df_ = df.groupby(ni).progress_apply(lambda x: cluster_optimally_pd(x[ye], 2))
        df = pd.merge(df, df_, how='left', left_index=True, right_index=True)
        print(df.head())
        print('*** value counts of {0}'.format(wi))
        print(df[wi].value_counts())
        print('*** value counts of {0}'.format(nw))
        print(df.drop_duplicates(ni)[nw].value_counts())
        columns_dict['cycle'] = [nw, wi, 'd0']

    columns_lists = [columns_dict[k] for k in feature_groups]
    columns = [x for sublist in columns_lists for x in sublist]
    columns += [ps]

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
    print('n_notify (update run progress every n_notify): {0}'.format(n_notify))
    print('present_columns: {0}'.format(columns))

    for idc in ids:
        data_dict[str(idc)] = df.loc[df[ni].isin([idc]), columns].values.T
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

    datatype = '_'.join(columns)

    str_to_hash = '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(df_type, version, datatype,
                                                       n_samples, low_bound_history_length,
                                                       low_freq, hi_freq)

    datatype_hash = int(sha1(str_to_hash.encode('utf-8')).hexdigest(), 16)
    datatype_hash_trunc = datatype_hash % 1000000

    if test_head < 0:
        with open(expanduser('~/data/kl/logs/generate_data_dict_runs.txt'), 'a') as f:
            f.write('{0} : {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8}\n'.format(datatype_hash_trunc,
                                                                                   df_type, version, datatype,
                                                                                   n_samples, low_bound_history_length,
                                                                                   low_freq, hi_freq, test_head))

    if test_head < 0:
        with gzip.open(expanduser('~/data/kl/batches/data_batches_{0}_v_{1}'
                                  '_hash_{2}.pgz'.format(df_type, version, datatype_hash_trunc)), 'wb') as fp:
            pickle.dump(data_batches, fp)

    if test_head < 0:
        fname = expanduser('~/data/kl/batches/df_{0}_v_{1}_hash_{2}.h5'.format(df_type, version, datatype_hash_trunc))
        store = pd.HDFStore(fname)
        store.put('df', df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
