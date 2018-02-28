import pandas as pd
import bm_support.gene_id_converter as bgc
import datahelpers.dftools as dfto
from os.path import expanduser
import numpy as np
from wos_parser.parse import issn2int
import pickle
import gzip
from datahelpers.aux import unfold_df, find_closest_year
from datahelpers.constants import pm, ye, ai, ps, up, dn, ar, ni

full_statement_flag = True

origin = 'lit'
version = 8

upstr = 'upstream'
dnstr = 'downstream'
rtype = 'Regulation Type'
sentence = 'Sentence.1'

df = pd.read_csv(expanduser('~/data/literome/pathway-extraction.txt.gz'), sep='\t', compression='gzip')

df.rename(columns={'PMID': pm}, inplace=True)

# to save pmids so that years and issn could be pulled later
pmids = df[pm].drop_duplicates()

pd.DataFrame(pmids.values, columns=[pm]).to_csv(expanduser('~/data/literome/literome_pmids.csv.gz'),
                                                sep=',', compression='gzip')

# cut out complexes '_', work with families '|'
mcut = (df['Theme'].str.contains('_')) | (df['Cause'].str.contains('_'))
print(float(sum(~mcut)) / mcut.shape[0])
df = df[~mcut]

df[up] = df['Cause'].apply(lambda x: x.split(':')[-1])
df[dn] = df['Theme'].apply(lambda x: x.split(':')[-1])
df[upstr] = df[up]
df[dnstr] = df[dn]

# define action type
m = (df[rtype] == 'Positive_regulation')
df[ps] = True
df.loc[~m, ps] = False

# expand families
cols = [up, dn, ps]
df_tmp = df[cols]
p, q = unfold_df(df_tmp)

# merge back pmids
dfs = pd.DataFrame(q, columns=(['index'] + list(p)))
print(dfs.shape)
dfi2 = pd.merge(dfs, df[[pm, rtype, sentence, upstr, dnstr]], left_on='index', right_index=True)
print('after pmid remerge:', dfi2.shape)

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

mps = 'mean_ps'
ps_mean = dfi2.groupby([up, dn, pm]).apply(lambda x: x[ps].mean())
dfi2_ = ps_mean.reset_index().rename(columns={0: mps})
dfi2_[ps] = 1.0
mask_indefinite = (dfi2_[mps] == 0.5)
dfi2_.loc[mask_indefinite, ps] = -1
mask_neg = (dfi2_[mps] < 0.5)
dfi2_.loc[mask_neg, ps] = 0.0

dfi2_ = dfi2_.merge(dfi2[[pm, up, dn, ps, rtype, sentence, upstr, dnstr]], on=[pm, up, dn, ps])

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
pmids2 = pd.merge(pd.DataFrame(pmids, columns=[pm]), df_pmid, how='inner', on=pm)
print('number of pmids dropped: {0}'.format(pmids.shape[0]-pmids2.shape[0]))

# merge (pm-issn) onto (claims)
dfi3 = pd.merge(dfi2_, pmids2[[pm, 'issn', 'year']], on=pm, how='left')
print('dfi3.shape: {0}'.format(dfi3.shape))

dfi3 = dfi3.loc[~dfi3[ye].isnull()].copy()
dfi3[pm] = dfi3[pm].astype(int)
dfi3[ye] = dfi3[ye].astype(int)
dfi3[ps] = dfi3[ps].astype(int)

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


df_affs = pd.read_csv(expanduser('~/data/tmp/aff_rating.csv.gz'),
                      compression='gzip').rename(columns={'rating': ar})

dfi5 = pd.merge(dfi4, df_affs, how='left', on=pm)

dfi5[ar] = dfi5[ar].fillna(-1)

dfi6 = dfto.get_multiplet_to_int_index(dfi5, [up, dn], ni)
print(dfi6[ai].value_counts().head())


with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(origin, version)), 'wb') as fp:
    pickle.dump(dfi6[[ni, pm, up, dn, ps, ye, ai, ar]], fp)

if full_statement_flag:
    with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}_fs.pgz'.format(origin, version)), 'wb') as fp:
        pickle.dump(dfi6[[ni, pm, up, dn, ps, ye, ai, ar, upstr, dnstr, rtype, sentence]], fp)

