import argparse
import pandas as pd
import numpy as np
from datahelpers.dftools import extract_idc_within_frequency_interval
from sklearn.preprocessing import MinMaxScaler
import gzip
import pickle
from os.path import expanduser
from datahelpers.sampling import split_to_subsamples

ye = 'year'
up = 'up'
dn = 'dn'
ni = 'new_index'
iden = 'identity'
idt = ni
ps = 'pos'
ai = 'ai'


def main(df_type, version, max_size, present_columns, transform_columns, a, b, n):

    with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(df_type, version)), 'rb') as fp:
        df = pickle.load(fp)

    if ai in present_columns:
        alpha = 0.9
        mask = (df[ai] > alpha)
        df[ai + 'hi'] = 0
        df.loc[mask, ai + 'hi'] = 1
        df[ai] = (df[ai] - alpha) / (1 - alpha)
        print(sum(mask), sum(~mask))
        df.loc[~mask, ai] = 0
        present_columns.insert(present_columns.index(ai) + 1, ai + 'hi')

    print(present_columns)

    if iden in present_columns:
        df[iden] = 1

    dft = df[[ni] + present_columns].copy()

    a = 0.1
    b = 0.9
    n = 20
    ids = extract_idc_within_frequency_interval(dft, ni, ps, (a, b), n)
    print('number of unique ids : {0}'.format(len(ids)))

    mask_ids = df[ni].isin(ids)
    means = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ps].mean())
    sizes = df.loc[mask_ids].groupby(ni).apply(lambda x: x[ps].shape[0])

    df_uniques = df.loc[mask_ids, [up, dn, ni]].drop_duplicates(ni)
    print(df_uniques.shape, len(df_uniques[ni].unique()))

    means_lens = pd.merge(pd.DataFrame(means, columns=['mean']), pd.DataFrame(sizes, columns=['len']),
                          left_index=True, right_index=True)

    df_stats = pd.merge(df_uniques, means_lens, right_index=True, left_on=ni).sort_values(['len'], ascending=False)

    df_stats.to_csv(expanduser('~/data/kl/claims/pairs_freq_{0}_v_{1}_n_{2}'
                               '_a_{3}_b_{4}.csv.gz'.format(df_type, version, n, a, b)),
                    compression='gzip', index=False)

    data_dict = {str(idc): dft.loc[dft[ni].isin([idc]), present_columns].values.T[:] for idc in ids}
    data_dict2 = {}
    for c in transform_columns:
        if c in present_columns:
            index = present_columns.index(c)
            for k, d in data_dict.items():
                sc = MinMaxScaler()
                d2 = d.copy().astype(np.float)
                d2[index] = np.squeeze(sc.fit_transform(d[index].reshape(-1, 1)))
                data_dict2[k]= d2

    metric_dict = {k: v.shape[1] for k, v in data_dict2.items()}

    idc_partition = split_to_subsamples(metric_dict, size=max_size)

    data_batches = [{k: data_dict2[k] for k in sub} for sub in idc_partition]

    datatype = '_'.join(present_columns)

    with gzip.open(expanduser('~/data/kl/batches/data_batches_{0}_v_{1}_c_{2}_m_{3}_'
                              'n_{4}_a_{5}_b_{6}.pgz'.format(df_type, version, datatype, max_size,
                                                             n, a, b)), 'wb') as fp:
        pickle.dump(data_batches, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # version = 8
    # version = 9

    # version = 1

    parser.add_argument('-d', '--datasource',
                        default='gw',
                        help='type of data to work with [gw, lit]')

    parser.add_argument('-v', '--version',
                        default=8, type=int,
                        help='version of data source')

    parser.add_argument('-s', '--batchsize',
                        default=1000, type=int,
                        help='size of data batches')

    parser.add_argument('-n', '--maxsize-sequence',
                        default=20, type=int,
                        help='version of data source')

    parser.add_argument('-p', '--partition-sequence',
                        nargs='+', default=[0.1, 0.9], type=float,
                        help='define interval of observed freqs for sequence consideration')

    # parser.add_argument('-d', '--destpath', default='.',
    #                     help='Folder to write data to, Default is current folder')

    parser.add_argument('--data-columns', nargs='*', default=['year', 'identity', 'ai', 'pos'])

    parser.add_argument('--transform-columns', nargs='*', default=['year'])

    args = parser.parse_args()

    print(args._get_kwargs())

    a, b = args.partition_sequence
    n = args.maxsize_sequence

    main(args.datasource, args.version, args.batchsize, args.data_columns,
         args.transform_columns, a, b, n)
