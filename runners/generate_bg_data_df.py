import pandas as pd
from os.path import expanduser
import json
import datahelpers.collapse as dc
import datahelpers.dftools as dfto
import datahelpers.sql as ds
from wos_parser.parse import issn2int
import wos_agg.aux as waa
import numpy as np
import pickle
import gzip

up_alias = 'Entrez Gene Interactor A'
dn_alias = 'Entrez Gene Interactor B'
at_str = 'Experimental System'
at_str_2 = 'Experimental System Type'
pmid_alias = 'Pubmed ID'
source = 'Source Database'
up = 'up'
dn = 'dn'
ps = 'pos'
at = 'action'
at = 'pos'
ni = 'new_index'
pm = 'pmid'
ye = 'year'
ai = 'ai'


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


def drop_duplicates_cols_arrange_col(dft, columns, col):
    # drop rows with col == 'NULL'
    # drop (ni, pm) duplicates
    # only max value of col remain from duplicates
    maskt = (dft[col] == 'NULL')
    print('fraction of claims with missing '
          'precision dropped: {0:.4f}'.format(float(sum(maskt)) / maskt.shape[0]))
    dft2 = dft.loc[~maskt].copy()
    dft2[col] = dft2[col].astype(float)
    dft2 = dft2.reset_index(drop=True)
    idx = dft2.groupby(columns)[col].idxmax()
    dft3 = dft2.loc[idx]
    # df3 = df3.drop_duplicates(columns)
    print('fraction of claims (same pmid extractions) dropped: {0:.4f}'.format(1. - float(df3.shape[0]) / df2.shape[0]))
    return dft3


df = pd.read_csv('/Users/belikov/data/biogrid/BIOGRID-ALL-3.4.152.tab2.zip', header=0,
                 sep='\t', compression='zip', low_memory=False)


df2 = df.rename(columns={up_alias: up, dn_alias: dn, pmid_alias: pm})[[up, dn, pm, at_str]].copy()


m1 = df2[up].apply(waa.is_int)
print(sum(m1), m1.shape[0])
m2 = df2[dn].apply(waa.is_int)
print(sum(m2), m2.shape[0])
df2 = df2.loc[m1 & m2]
df2[up] = df2[up].astype(int)
df2[dn] = df2[dn].astype(int)

orig = 'bg'
with open(expanduser('~/data/kl/claims/actions_{0}.json'.format(orig)), 'r') as fp:
    agg_act_dict = json.load(fp)

bools = {'true': True, 'false': False}

agg_act_dict = {bools[k]: v for k, v in agg_act_dict.items()}

invdd = dc.invert_dict_of_list(agg_act_dict)


df3 = dfto.attach_new_index(df2, invdd, [at_str, at], [up, dn], ni)

print(df3[at].value_counts())
print(df3[at_str].value_counts())

pmids = df3[pm].drop_duplicates()
print(pmids.shape)

# create the query dict
query_doc = {'columns': ['pmid', 'issn', 'year'], 'mask': {'pmid': []}}

query_aff = {'columns': ['pmid', 'affiliation'], 'mask': {'pmid': [9988722, 3141384, 1924363]}}

pmids_list = list(pmids)[:]

# create the query dict
query_doc = {'columns': [pm, 'issn', 'year'], 'mask': {pm: []}}

query_aff = {'columns': [pm, 'affiliation'], 'mask': {pm: []}}

pmids_list = list(pmids)[:]
query_doc['mask'][pm] = pmids_list

# qq = {'doc': query_doc, 'affiliation': query_aff}
qq = {'doc': query_doc}

dfs = {}
for k, v in qq.items():
    dfs[k] = ds.get_table('a', database='medline', table=k,
                          query_dict=v)
df_pmid = dfs['doc']
mask_issn = df_pmid['issn'].notnull()
print(sum(mask_issn)/mask_issn.shape, sum(mask_issn), mask_issn.shape)
df_pmid['issn_str'] = df_pmid['issn']
df_pmid.loc[mask_issn, 'issn'] = df_pmid.loc[mask_issn,
                                             'issn'].apply(issn2int)


df_pmid = df_pmid.loc[~df_pmid['year'].isnull()]
df_pmid['year'] = df_pmid['year'].astype(int)
print(df_pmid.shape)


dfi3 = pd.merge(df_pmid, df3, how='left', on=pm)

set_pmids_issns = set(df_pmid['issn'].unique())

# retrieve and merge issn-ye-ef-ai table (issn-ye-ai)
df_ai = pd.read_csv(expanduser('~/data/kl/eigen/ef_ai_1990_2014.csv.gz'), index_col=0, compression='gzip')

set_ai_issns = set(df_ai['issn'].unique())
print('{0} issns in pmids-issn table that are not ai table'.format(len(set_pmids_issns - set_ai_issns)))
print('{0} issns in pmids-issn table that are in ai table'.format(len(set_pmids_issns & set_ai_issns)))
working_pmids = set(dfi3['pmid'].unique())
issn_pmids = set(df_pmid['pmid'].unique())
print('{0} of pmids from biogrid are not in pmid-issn table'.format(len(working_pmids - issn_pmids)))
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

print(dfi4[ai].value_counts().head())

dfi6 = dfi4.copy()
dfi6 = dfi6[[ni, up, dn, at, ye, ai]]


with gzip.open(expanduser('~/data/kl/claims/df_bg_1.pgz'), 'wb') as fp:
    pickle.dump(dfi6, fp)

