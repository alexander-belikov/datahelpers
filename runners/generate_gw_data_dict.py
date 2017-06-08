import pandas as pd
import numpy as np
import datahelpers.dftools as dfto
from sklearn.preprocessing import MinMaxScaler
import gzip
import pickle
from os.path import expanduser

pm = 'pmid'
ii = 'intId'
ye = 'year'
iss = 'issn'
up = 'upstream'
dn = 'downstream'
upe = 'up'
dne = 'dn'
rt = 'regulation_type'
ng = 'negative'
dict_rt = {'Positive_regulation': 1, 'Negative_regulation': 0,
           'Unspecified': np.nan, np.nan: np.nan,
           'Activation': 1, 'Inhibition': 0}
at = 'at2'
ni = 'new_index'
nclaims = 'actionmention_count'
fr = 'frac'
am = 'ambi'
iden = 'identity'
idt = ni
ps = 'pos'
pr = 'prec'
ai = 'ai'
ai_cdf = 'ai_cdf'
hi_ai = 'hiai'
ti = 'time'
version = 8

with gzip.open(expanduser('~/data/kl/claims/df_cs_{0}.pgz'.format(version)), 'rb') as fp:
    df = pickle.load(fp)

transform_time = True

df2 = df[[ni, pm, at, ai_cdf, ps, pr, ye, upe, dne]].copy()

alpha = 0.9
mask = (df2[ai_cdf] > alpha)
df2[hi_ai] = 0
df2.loc[mask, hi_ai] = 1
df2[ai] = (df2[ai_cdf] - alpha)/(1 - alpha)
print(sum(mask), sum(~mask))
df2.loc[~mask, ai] = 0
df2[iden] = 1

dft = df2[[ni, at, ps, ye, ai, hi_ai, iden, pr, pm, upe, dne]].copy()

a = 0.1
b = 0.9
n = 20
ids = dfto.extract_idc_within_frequency_interval(dft, ni, ps, (a, b), n)
print('number of unique ids : {0}'.format(len(ids)))

mask_ids = dft[ni].isin(ids)
means = dft.loc[mask_ids].groupby(ni).apply(lambda x: x[ps].mean())
sizes = dft.loc[mask_ids].groupby(ni).apply(lambda x: x[ps].shape[0])

df_uniques = dft.loc[mask_ids, [upe, dne, ni, at]].drop_duplicates(ni)
print(df_uniques.shape, len(df_uniques[ni].unique()))
means_lens = pd.merge(pd.DataFrame(means, columns=['mean']), pd.DataFrame(sizes, columns=['len']),
                      left_index=True, right_index=True)

df_stats = pd.merge(df_uniques, means_lens, right_index=True, left_on=ni).sort_values(['len'], ascending=False)

df_stats.to_csv(expanduser('~/data/kl/claims/gw_pairs_freq_v_{0}_n_{1}_a_{2}_b_{3}.csv.gz'.format(version, n, a, b)),
                compression='gzip', index=False)

ids_props = {}
feauture_cols = [ai, hi_ai]

for ii in ids:
    df_cut = dft.loc[dft[ni].isin([ii])]
    ii_len = df_cut.shape[0]
    means = {'mean_{0}'.format(c): df_cut[c].mean() for c in feauture_cols}
    stds = {'std_{0}'.format(c): df_cut[c].std() for c in feauture_cols}
    mins = {'min_{0}'.format(c): df_cut[c].min() for c in feauture_cols}
    maxs = {'max_{0}'.format(c): df_cut[c].max() for c in feauture_cols}
    left = {'left_{0}'.format(c): means['mean_{0}'.format(c)] - stds['std_{0}'.format(c)] for c in feauture_cols}
    right = {'right_{0}'.format(c): means['mean_{0}'.format(c)] + stds['std_{0}'.format(c)] for c in feauture_cols}
    densities = {'den_{0}'.format(c): 2*stds['std_{0}'.format(c)]/ii_len for c in feauture_cols}
    ids_props[ii] = {'len': ii_len}
    ids_props[ii].update(means)
    ids_props[ii].update(stds)
    ids_props[ii].update(densities)
    ids_props[ii].update(left)
    ids_props[ii].update(right)
    ids_props[ii].update(mins)
    ids_props[ii].update(maxs)


sampling_df = pd.DataFrame(ids_props).T
sampling_df.head()

sdf = sampling_df.copy()
sdf = sdf.sort_values('len', ascending=False)

lr_cols = ['left_ai', 'right_ai']
sdf_ai = sdf.sort_values(lr_cols, ascending=False)

max_size = 1000
batches = []
batches_lens = []
lrs = []

while sdf.shape[0] > 0:
    batch = []
    batch_len = []
    while sum(batch_len) < max_size and sdf.shape[0] > 0:
        if batch:
            sdf_tmps = sdf_ai.copy()
            sdf_tmps[lr_cols[0]] = sdf_tmps[lr_cols[0]].apply(lambda x: max(left - x, 0))
            sdf_tmps[lr_cols[1]] = sdf_tmps[lr_cols[1]].apply(lambda x: max(x - right, 0))
            sdf_tmps.sort_values(lr_cols, ascending=False)

            row = sdf_tmps.iloc[0]            
            id_ = row.name
            batch.append(id_)
            batch_len.append(row['len'])
            if left > row['left_ai']:
                left = row['left_ai']
            if right < row['right_ai']:
                right = row['right_ai']
            sdf.drop(id_, inplace=True)
            sdf_ai.drop(id_, inplace=True)
        else:
            row = sdf.iloc[0]
            id_ = row.name
            batch.append(id_)
            batch_len.append(row['len'])
            left = row['left_ai']
            right = row['right_ai']
            sdf.drop(id_, inplace=True)
            sdf_ai.drop(id_, inplace=True)
    if sdf.shape[0] == 0:
        batches[-1].extend(batch)
        batches_lens[-1] += sum(batch_len)
    else:
        batches.append(batch)
        batches_lens.append(sum(batch_len))
        lrs.append((left, right))


data_batches = []

# feauture_cols = [hi_ai]
# feauture_cols = [ai, hi_ai, pr]
feauture_cols = [ai, hi_ai]

# important_cols = [iden] + feauture_cols + [ps]
important_cols = [ye, iden] + feauture_cols + [ps]

for batch in batches:
    data_dict = {str(idc): dft.loc[dft[ni].isin([idc]), important_cols].values.T[:] for idc in batch}
    data_dict2 = {}
    if transform_time:
        for k, d in data_dict.items():
            sc = MinMaxScaler()
            d2 = d.copy()
            d2[0] = np.squeeze(sc.fit_transform(d[0].astype(np.float).reshape(-1, 1)))
            data_dict2.update({k: d2})
   
    data_batches.append(data_dict2)


with gzip.open(expanduser('~/data/kl/batches/data_batches_v_{0}_c_{1}_m_{2}_'
                          'n_{3}_a_{4}_b_{5}.pgz'.format(version, '_'.join(important_cols), max_size,
                                                         n, a, b)), 'wb') as fp:
    pickle.dump(data_batches, fp)
