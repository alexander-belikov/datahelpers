import pandas as pd
import bm_support.gene_id_converter as bgc
import datahelpers.dftools as dfto
from os.path import expanduser
from itertools import product
import numpy as np
from wos_parser.parse import issn2int
from datahelpers.aux import unfold_df, find_closest_year
import pickle
import gzip

up = 'up'
dn = 'dn'
ps = 'pos'
at = 'action'
at = 'pos'
ni = 'new_index'
pm = 'pmid'
ye = 'year'
ai = 'ai'
pm = 'pmid'
version = 7

df = pd.read_csv(expanduser('/Users/belikov/data/literome/literome_fulltext_claims.csv.gz'),
                 sep=',', compression='gzip')

df.rename(columns={'Pmid': pm}, inplace=True)

pmids = df[pm].drop_duplicates()


conv_dict = {'Positive_regulation': True, 'Negative_regulation': False}


df2 = df.loc[df['EventType'].isin(conv_dict.keys())]

df2[at] = df2['EventType'].apply(lambda x: conv_dict[x])


df3 = df2.rename(columns={'Cause':up, 'Theme': dn})


# convert symbols to entrez id
gc = bgc.GeneIdConverter(expanduser('~/data/chebi/hgnc_complete_set.json.gz'), bgc.types, bgc.enforce_ints)
gc.choose_converter('symbol', 'entrez_id')

set_symbols = gc.convs['symbol', 'entrez_id']
m_up = df3[up].isin(set_symbols)
m_dn = df3[dn].isin(set_symbols)
df4 = df3[m_up & m_dn].copy()

gc.choose_converter('symbol', 'entrez_id')
df4[up] = df4[up].apply(lambda x: gc[x])
df4[dn] = df4[dn].apply(lambda x: gc[x])


# multiple mentions from the same paper: we have to identify the vote!

df5 = df4[[up, dn, ps, pm]].groupby([up, dn, pm]).apply(lambda x: x[ps].mean())


df6 = df4[[up, dn, ps, pm]].groupby([up, dn, pm]).apply(lambda x: x[ps].shape[0])

# study overlap between dfi2 (abstract) and df4[[up, dn, ps, pm]].drop_duplicates([up, dn, pm]) (fulltext)
df_ff = df4[[up, dn, pm]].drop_duplicates([up, dn, pm])
print(df_ff.shape)


undecidables_mask = (df5 > 0.4) & (df5 < 0.6)
decidables = df5[~undecidables_mask].copy()


decidables_bool = decidables.apply(lambda x: True if x > 0.5 else False)


decidables = decidables_bool.reset_index().rename(columns={0:ps})


dfi2 = decidables


with gzip.open(expanduser('~/data/kl/raw/medline_doc_cs_5.pgz'), 'rb') as fp:
    df_pmid = pickle.load(fp)

mask_issn = df_pmid['issn'].notnull()
df_pmid['issn_str'] = df_pmid['issn']
df_pmid.loc[mask_issn, 'issn'] = df_pmid.loc[mask_issn, 'issn'].apply(issn2int)

# drop pmids without years
df_pmid = df_pmid.loc[~df_pmid['year'].isnull()]

# convert years to int
df_pmid['year'] = df_pmid['year'].astype(int)

# merge literome pmids to
pmids2 = pd.merge(pd.DataFrame(pmids, columns=[pm]), df_pmid, how='inner', on=pm)
print('number of pmids dropped: {0}'.format(pmids.shape[0]-pmids2.shape[0]))


# merge (pm-issn) onto (claims)
dfi3 = pd.merge(dfi2, pmids2[[pm, 'issn', 'year']], on=pm, how='left')
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
print('{0} ai value imputed, out of {1}. It is {2:.3f} fraction'.format(sum(mask), mask.shape[0], sum(mask)/mask.shape[0]))

dfi5 = dfto.get_multiplet_to_int_index(dfi4, [up, dn], ni)
print(dfi5[ai].value_counts().head())

dfi6 = dfi5.copy()
dfi6 = dfi6[[ni, up, dn, at, ye, ai]]


with gzip.open(expanduser('~/data/kl/claims/df_lit_{0}.pgz'.format(version)), 'wb') as fp:
    pickle.dump(dfi6, fp)



