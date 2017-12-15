import numpy as np
import pandas as pd
import datahelpers.collapse as dc
import datahelpers.dftools as dfto
from wos_parser.parse import issn2int
import pickle
import gzip
import json
from os.path import expanduser
from datahelpers.aux import find_closest_year, drop_duplicates_cols_arrange_col

# TODO revamp min year for ef_ai
# TODO attach ai ~ separate func

with gzip.open(expanduser('~/data/kl/raw/val_geneways_cs_0.pgz'), 'rb') as fp:
    dfi = pickle.load(fp)

hi = 'hiid'
up = 'upstream'
dn = 'downstream'
up2 = 'up'
dn2 = 'dn'
am_id = 'actionmentionid'
ft = 'isFullText'
at = 'at2'
ng = 'negative'
sn = 'sentencenumber'
at_orig = 'actiontype'
pm = 'pmid'
ye = 'year'
sc = 'score'
pr = 'prec'
eat = 'exp_at'
gt = 'gg'
ps = 'pos'
aiexp = 'ArticleInfluence'
ni = 'new_index'
origin = 'gw'
ai = 'ai'
ar = 'ar'
# version = 8
# version = 9
# version with affiliation ranking
version = 10

version2actions = {8: '', 9: '_v2', 10: ''}

print('working to produce version {0}'.format(version))

dfdd = {}
dfi2, dfdd = dc.collapse_df(dfi, str_dicts=dfdd, dropna_columns=[pm, hi],
                            bool_columns=[ng, ft],
                            numeric_columns=[hi, sn, am_id, pm, sc, pr])

pd.DataFrame(dfi2[pm].unique(), columns=[pm]).to_csv(expanduser('~/data/gw/gw_pmids.csv.gz'),
                                                     sep=',', compression='gzip')

with gzip.open(expanduser('~/data/kl/raw/medline_doc_cs_2.pgz'), 'rb') as fp:
    df_pmid = pickle.load(fp)

df_pmid = df_pmid[~df_pmid['issn'].isnull()]

df_pmid['issn_str'] = df_pmid['issn']
df_pmid['issn'] = df_pmid['issn'].apply(issn2int)

# drop pmids without years
df_pmid = df_pmid.loc[~df_pmid['year'].isnull()]

# convert years to int
df_pmid['year'] = df_pmid['year'].astype(int)
set_pmids_issns = set(df_pmid['issn'].unique())

# retrieve issn-ye-ef-ai table (issn-ye-ai)
df_ai = pd.read_csv(expanduser('~/data/kl/eigen/ef_ai_1990_2014.csv.gz'), index_col=0, compression='gzip')

set_ai_issns = set(df_ai['issn'].unique())
print('{0} issns in pmids-issn table that are not ai table'.format(len(set_pmids_issns - set_ai_issns)))
print('{0} issns in pmids-issn table that are ai table'.format(len(set_pmids_issns & set_ai_issns)))
working_pmids = set(dfi2['pmid'].unique())
issn_pmids = set(df_pmid['pmid'].unique())
print('{0} of pmids from geneways that are not in pmid-issn table'.format(len(working_pmids - issn_pmids)))
mask = df_pmid['issn'].isin(list(set_ai_issns))
print('{0} of pmids in pmid-issn table that are in issn-ai table'.format(sum(mask)))


# cut (pm-issn) to issns only in (issn-ye-aiai)
df_pmid2 = df_pmid.loc[mask]
dfi3 = pd.merge(dfi2, df_pmid2, on='pmid')
# merge (pm-issn) onto (claims)

fraction = float(dfi3.shape[0])/dfi2.shape[0]
print('fraction of claims with issn {0:.4f}'.format(fraction))

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
df_proxy_years = pd.DataFrame(np.array(list_proxy_year), columns=['issn', 'year', 'proxy_year'])

# merge (pm-issn-ye) join (issn-ye-ai) onto (claims-pm)
df_pmid3 = pd.merge(df_pmid2, df_proxy_years, on=['issn', 'year'])
df_ai = df_ai.rename(columns={'year': 'ai_year'})
df_feature = pd.merge(df_pmid3, df_ai, left_on=['issn', 'proxy_year'], right_on=['issn', 'ai_year'])
df_feature_cut = df_feature[['pmid', 'ai_cdf']].rename(columns={'ai_cdf': ai})
dfi4 = pd.merge(dfi3, df_feature_cut, on=pm)

df_affs = pd.read_csv(expanduser('~/data/tmp/aff_rating.csv.gz'),
                      compression='gzip').rename(columns={'rating': ar})

dfi4 = pd.merge(dfi4, df_affs, how='left', on=pm)

dfi4[ar] = dfi4[ar].fillna(-1)

# merge (id-pm-claims-issn-ye-ai) onto (id-up-dn)
df_ha = pd.read_csv('~/data/kl/raw/human_action.txt.gz',
                    sep='\t', index_col=None, compression='gzip')

# version 8 uses actions.json
# version 9 usues actions_v2.json
with open(expanduser('~/data/kl/claims/actions{0}.json'.format(version2actions[version])), 'r') as fp:
    agg_act_dict = json.load(fp)

bools = {'true': True, 'false': False}

agg_act_dict = {bools[k]: v for k, v in agg_act_dict.items()}

invdd = dc.invert_dict_of_list(agg_act_dict)

df_ni = dfto.attach_new_index(df_ha, invdd, [at_orig, at], [up2, dn2], ni)
dfi5 = dfi4.merge(df_ni[[hi, ni, up2, dn2, at]], on=hi)
fraction = float(dfi5.shape[0])/dfi4.shape[0]
print('fraction of claims remaining after human action merge and boolean '
      'action aggregation: {0:.4f}'.format(fraction))
# in case there are degenerate extractions from the same pmid, leave only claims with maximum precision
dfi6 = drop_duplicates_cols_arrange_col(dfi5, [ni, pm], pr)
fraction = float(dfi6.shape[0])/dfi5.shape[0]
print('fraction of claims remaining after multiple similar claims from the same pmid: {0:.4f}'.format(fraction))
fraction = sum(dfi6['year'] < 1990)/dfi6.shape[0]
print('fraction of claims before 1990: {0:.4f}'.format(fraction))

# create positive claim column
# new_index is defined up, dn pair; 
# the actions can as positive and negative
# by default the action is deemed positive
# the claim
# ps is logical xor between at and ng
# ps = (at | ng) ^ (at & ng)

def xor(df, ccs, c):
    c1, c2 = ccs
    print('number of {0} rows: {1}'.format(c1, sum(df[c1])))
    print('number of {0} rows: {1}'.format(c2, sum(df[c2])))
    print('{0}'.format(list(zip(ccs, df[ccs].dtypes))))
    df2 = df.copy()
    mask_negs = ~(df2[c1])
    df2[c] = ~df2[c2]
    df2.loc[mask_negs, c] = df2.loc[mask_negs, c2]
    df2[c] = df2[c].astype(int)
    print('number of negative claims on reduced statements {0}'.format(df2.shape[0] - sum(df2[c])))
    return df2

dfi7 = xor(dfi6, [at, ng], ps)

print('number of rows in saved df_{0}_{1}.pgz: {2}'.format(origin, version, dfi7.shape[0]))


# convention to conform with literome
# TODO fix columns' names

at = ps
up = up2
dn = dn2

dfi7 = dfi7[[ni, up, dn, at, ye, ai, ar]]

with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(origin, version)), 'wb') as fp:
    pickle.dump(dfi7, fp)
