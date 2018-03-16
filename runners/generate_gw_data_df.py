import numpy as np
import pandas as pd
import datahelpers.collapse as dc
import datahelpers.dftools as dfto
from wos_parser.parse import issn2int
import pickle
import gzip
import json
from os.path import expanduser
from datahelpers.aux import find_closest_year
from datahelpers.constants import pm, ye, ai, ps, ar, ni, up, dn


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

# TODO revamp min year for ef_ai
# TODO attach ai ~ separate func

origin = 'gw'
# version = 8
# version = 9
# version with affiliation ranking
version = 10
version = 11

full_statement_flag = True

hi = 'hiid'
ng = 'negative'
at_orig = 'actiontype'
ft = 'isFullText'
at = 'act'
upstr = 'upstream'
dnstr = 'downstream'

with gzip.open(expanduser('~/data/kl/raw/val_geneways_cs_0.pgz'), 'rb') as fp:
    dfi = pickle.load(fp)

# leave only claims from abstracts
dfi2 = dfi.loc[dfi[ft] == 'N', [hi, ng, pm, at_orig, upstr, dnstr]]

version2actions = {8: '', 9: '_v2', 10: '', 11: ''}


dfdd = {}
dfi2, dfdd = dc.collapse_df(dfi2, str_dicts=dfdd, dropna_columns=[pm, hi],
                            bool_columns=[ng],
                            numeric_columns=[pm, hi])

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

df_ni = dfto.attach_new_index(df_ha, invdd, [at_orig, at], [up, dn], ni)

dfi2 = dfi2.merge(df_ni[[hi, ni, up, dn, at]], on=hi)

dfi2 = xor(dfi2, [at, ng], ps)

unambiguous_extraction = dfi2.groupby([ni, pm]).apply(lambda x: len(x[ps].unique())).reset_index()
good_claims = unambiguous_extraction.loc[unambiguous_extraction[0] == 1, [ni, pm]]
dfi2 = dfi2.merge(good_claims, on=[ni, pm], how='right').drop_duplicates([ni, pm])

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

df_affs = df_affs.drop_duplicates(pm)

dfi4 = pd.merge(dfi4, df_affs, how='left', on=pm)

dfi4[ar] = dfi4[ar].fillna(-1)

with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(origin, version)), 'wb') as fp:
    pickle.dump(dfi4[[ni, pm, up, dn, ps, ye, ai, ar]], fp)

if full_statement_flag:
    with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}_fs.pgz'.format(origin, version)), 'wb') as fp:
        pickle.dump(dfi4[[ni, pm, up, dn, ps, ye, ai, ar, upstr, dnstr, at_orig, ng]], fp)
