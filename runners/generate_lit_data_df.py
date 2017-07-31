import pandas as pd
import bm_support.gene_id_converter as bgc
import datahelpers.dftools as dfto
from os.path import expanduser
from itertools import product
import numpy as np
from wos_parser.parse import issn2int
import pickle
import gzip


def cut_string(x):
    if '_' in x:
        x2 = x.split('_')
        if '|' in x:
            # if we need hgnc ids, work with x3
            x3 = x2[0].split('|')
            r = x2[1:]
        else:
            r = x2[1:]
    else:
        r = [x]
    return r


def unfold_df(df):
    acc = []
    for i, row in df.iterrows():
        up_, dn_, act = row.values
        x, y = cut_string(up_), cut_string(dn_)
        acc.extend(list(product(*[[i], x, y, [act]])))
    return df.columns, acc


def find_closest_year(x, years):
    # what if we don't the feature (article influence, e.g.) for a give (issn, year) pair?
    # yield the closet year for the same issn!
    # add proxy_years to (pm-issn-ye) which are closest to issn-ye in (issn-ye-ai)
    # years is sorted
    left = 0
    right = len(years) - 1
    if x <= years[left]:
        return years[left]
    elif x >= years[right]:
        return years[right]
    else:
        while right - left > 1:
            mid = (right + left) // 2
            if (x - years[left]) * (years[mid] - x) > 0:
                right = mid
            else:
                left = mid
        return years[left]


def drop_duplicates_cols_arrange_col(df, columns, col):
    # drop rows with col == 'NULL'
    # drop (ni, pm) duplicates
    # only max value of col remain from duplicates
    m = (df[col] == 'NULL')
    print('fraction of claims with missing precision dropped: {0:.4f}'.format(float(sum(m)) / m.shape[0]))
    df2 = df.loc[~m].copy()
    df2[col] = df2[col].astype(float)
    df2 = df2.reset_index(drop=True)
    idx = df2.groupby(columns)[col].idxmax()
    df3 = df2.loc[idx]
    # df3 = df3.drop_duplicates(columns)
    print('fraction of claims (same pmid extractions) dropped: {0:.4f}'.format(1. - float(df3.shape[0]) / df2.shape[0]))
    return df3


up = 'up'
dn = 'dn'
ps = 'pos'
at = 'action'
at = 'pos'
ni = 'new_index'
pm = 'pmid'
ye = 'year'
ai = 'ai'

df = pd.read_csv(expanduser('~/data/literome/pathway-extraction.txt.gz'), sep='\t', compression='gzip')

df.rename(columns={'PMID': pm}, inplace=True)

# to save pmids so that years and issn could be pulled later
pmids = df[pm].drop_duplicates()
pmids2 = pd.DataFrame(pmids.values, index=pmids.index, columns=[pm]).reset_index()

# cut out complexes '_', work with families '|'
mcut = (df['Theme'].str.contains('_')) | (df['Cause'].str.contains('_'))
print(float(sum(~mcut)) / mcut.shape[0])
df = df[~mcut]

df[up] = df['Cause'].apply(lambda x: x.split(':')[-1])
df[dn] = df['Theme'].apply(lambda x: x.split(':')[-1])

# define action type
m = (df['Regulation Type'] == 'Positive_regulation')
df[at] = True
df.loc[~m, at] = False

# expand families
cols = [up, dn, at]
df_tmp = df[cols]
p, q = unfold_df(df_tmp)

# merge back pmids
dfs = pd.DataFrame(q, columns=(['index'] + list(p)))
print(dfs.shape)
dfi2 = pd.merge(dfs, pd.DataFrame(df[pm]), left_on='index', right_index=True)
print(dfi2.shape)

# convert symbols to entrez id
gc = bgc.GeneIdConverter(expanduser('~/data/chebi/hgnc_complete_set.json.gz'), bgc.types, bgc.enforce_ints)
gc.choose_converter('symbol', 'entrez_id')

set_symbols = gc.convs['symbol', 'entrez_id']
m_up = dfi2[up].isin(set_symbols)
m_dn = dfi2[dn].isin(set_symbols)
dfi2 = dfi2[m_up & m_dn].copy()

gc.choose_converter('symbol', 'entrez_id')
dfi2[up] = dfi2[up].apply(lambda x: gc[x])
dfi2[dn] = dfi2[dn].apply(lambda x: gc[x])

###
# lookup pmids from medline and merge years
with gzip.open(expanduser('~/data/kl/raw/medline_doc_cs_4.pgz'), 'rb') as fp:
    df_pmid = pickle.load(fp)

mask_issn = df_pmid['issn'].notnull()
df_pmid['issn_str'] = df_pmid['issn']
df_pmid.loc[mask_issn, 'issn'] = df_pmid.loc[mask_issn, 'issn'].apply(issn2int)

# drop pmids without years
df_pmid = df_pmid.loc[~df_pmid['year'].isnull()]

# convert years to int
df_pmid['year'] = df_pmid['year'].astype(int)

# merge literome pmids to
pmids3 = pd.merge(pmids2, df_pmid, how='inner', on=pm)

# merge (pm-issn) onto (claims)
dfi3 = pd.merge(dfi2, pmids3[['index', 'issn', 'year']], on='index', how='left')
print('dfi3.shape: {0}'.format(dfi3.shape))

dfi3 = dfi3.loc[~dfi3[ye].isnull()].copy()
dfi3[pm] = dfi3[pm].astype(int)
dfi3[ye] = dfi3[ye].astype(int)
dfi3[at] = dfi3[at].astype(int)

set_pmids_issns = set(df_pmid['issn'].unique())

###
# retrieve and merge issn-ye-ef-ai table (issn-ye-ai)
df_ai = pd.read_csv(expanduser('~/data/kl/eigen/ef_ai_1990_2014.csv.gz'), index_col=0, compression='gzip')

set_ai_issns = set(df_ai['issn'].unique())
print('{0} issns in pmids-issn table that are not ai table'.format(len(set_pmids_issns - set_ai_issns)))
print('{0} issns in pmids-issn table that are ai table'.format(len(set_pmids_issns & set_ai_issns)))
working_pmids = set(dfi3['pmid'].unique())
issn_pmids = set(df_pmid['pmid'].unique())
print('{0} of pmids from literome are not in pmid-issn table'.format(len(working_pmids - issn_pmids)))
mask = df_pmid['issn'].isin(list(set_ai_issns))
print('{0} of pmids in pmid-issn table that are in issn-ai table'.format(sum(mask)))

# cut (pm-issn) to issns only in (issn-ye-aiai)
df_pmid2 = df_pmid.loc[mask]

df_pmid_reduced = df_pmid2[['issn', 'year']].drop_duplicates(['issn', 'year'])

dd_ai = {}
for it in df_ai[['issn', 'year']].iterrows():
    if it[1]['issn'] in dd_ai.keys():
        dd_ai[it[1]['issn']].append(it[1]['year'])
    else:
        dd_ai[it[1]['issn']] = [it[1]['year']]

list_proxy_year = []
for it in df_pmid_reduced.iterrows():
    ind, val = it
    proxy = find_closest_year(val['year'], dd_ai[val['issn']])
    list_proxy_year.append((val['issn'], val['year'], proxy))
# create issn, year (from literome), year (closest year from df_ai)
df_proxy_years = pd.DataFrame(np.array(list_proxy_year), columns=['issn', 'year', 'proxy_year'])

# merge (pm-issn-ye) onto (issn-ye-ai) onto (claims-pm)
df_pmid3 = pd.merge(df_pmid2, df_proxy_years, on=['issn', 'year'])
df_ai = df_ai.rename(columns={'year': 'ai_year'})
df_feature = pd.merge(df_pmid3, df_ai, left_on=['issn', 'proxy_year'], right_on=['issn', 'ai_year'])
df_feature_cut = df_feature[['pmid', 'ai_cdf']].rename(columns={'ai_cdf': 'ai'})
dfi4 = pd.merge(dfi3, df_feature_cut, on=pm, how='left')
print('dfi4.shape: {0}'.format(dfi4.shape))

# impute missing ai's with 0.5
mask = (dfi4[ai].isnull())
mean_available_ai = round(dfi4.loc[~mask, ai].mean(), 2)
print(mean_available_ai)
print(dfi4[ai].value_counts().head())
dfi4.loc[mask, ai] = mean_available_ai
print('{0} ai value imputed, out of {1}. It is {2:.3f}'.format(sum(mask), mask.shape[0], sum(mask)/mask.shape[0]))

dfi5 = dfto.get_multiplet_to_int_index(dfi4, [up, dn], ni)
print(dfi5[ai].value_counts().head())

dfi6 = dfi5.copy()
dfi6 = dfi6[[ni, up, dn, at, ye, ai]]

with gzip.open(expanduser('~/data/kl/claims/df_lit_5.pgz'), 'wb') as fp:
    pickle.dump(dfi6, fp)
