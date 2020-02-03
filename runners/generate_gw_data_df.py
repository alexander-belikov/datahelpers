from os.path import expanduser
import pandas as pd
import pickle
import gzip
import json
import datahelpers.collapse as dc
from datahelpers.constants import pm, ye, ai, ps, ar, ni, up, dn
from datahelpers.dftools import collapse_df, collapse_column

import numpy as np
import datahelpers.dftools as dfto
from wos_parser.parse import issn2int
from datahelpers.aux import find_closest_year
import seaborn as sns

hi = 'hiid'
ng = 'negative'
at_orig = 'actiontype'
ft = 'isFullText'
at = 'act'
sc = 'score'
prec = 'prec'
upstr = 'upstream'
dnstr = 'downstream'
from_abstract = 'from_abstract'
sn_bint = 'bin_int'


def print_stats(df):
    a = df.drop_duplicates([up, dn]).shape[0]
    b = df.drop_duplicates([up, dn, pm]).shape[0]
    c = df.drop_duplicates([pm]).shape[0]
    d = df.shape[0]
    print(f'unique up->dn pairs {a};\n'
          f'unique (up, dn, pm) {b};\n'
          f'unique pm {c};\n'
          f'unique rows {d}')


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


origin = 'gw'
# negation of negative goes to 0.75, negation of positive to 0.25
version = 12
# negation of negative goes to 1.0, negation of positive to 0.0
version = 13


dfi = pd.read_pickle('~/data/kl/raw/val_geneways_cs_0.pgz',
                     compression='gzip')

df_ha = pd.read_csv('~/data/kl/raw/human_action.txt.gz',
                    sep='\t', index_col=None, compression='gzip')

df_pmid = pd.read_pickle('~/data/kl/raw/medline_doc_cs_2.pgz',
                         compression='gzip')

df_affs = pd.read_csv('~/data/kotta/affiliations/aff_rating.csv.gz',
                      compression='gzip').rename(columns={'rating': ar})

# retrieve issn-ye-ef-ai table (issn-ye-ai)
df_ai = pd.read_csv(expanduser('~/data/kl/eigen/ef_ai_1990_2014.csv.gz'),
                    index_col=0, compression='gzip')


dfi.replace({sc: 'NULL', prec: 'NULL', pm: 'NULL'}, np.nan, inplace=True)
dfi[hi] = dfi[hi].astype(int)

dfi[sc]  = dfi[sc].astype(float)
dfi[prec]  = dfi[prec].astype(float)
dfi = dfi.loc[dfi[pm].notnull()]
dfi[pm] = dfi[pm].astype(int)

df_ha[hi] = df_ha[hi].astype(int)
dfi[ng] = (dfi[ng] == '1')

dfi[from_abstract] = (dfi[ft] == 'N')

version2actions = {8: '', 9: '_v2', 10: '', 11: '', 12: '', 13: '' }

with open(expanduser('~/data/kl/claims/actions{0}.json'.format(version2actions[version])), 'r') as fp:
    agg_act_dict = json.load(fp)

bools = {'true': True, 'false': False}

agg_act_dict = {bools[k]: v for k, v in agg_act_dict.items()}

invdd = dc.invert_dict_of_list(agg_act_dict)

df_ha[at] = df_ha[at_orig].apply(lambda x: invdd[x] if x in invdd.keys() else np.nan)


dfi_important_columns = [hi, pm, sc, prec, ng, from_abstract]

ha_important_columns = [hi, up, dn, at_orig, 'plo', at]

dfr = pd.merge(dfi[dfi_important_columns],
               df_ha[ha_important_columns], on=hi)

mask_pos = (dfr[at].notnull()) & (dfr[at] & (dfr[ng] == False))
mask_npos = (dfr[at].notnull()) & (dfr[at] & dfr[ng])

mask_neg = (dfr[at].notnull()) & ((dfr[at] == False) & (dfr[ng] == False))
mask_nneg = (dfr[at].notnull()) & ((dfr[at] == False) & dfr[ng])

dfr[sn_bint] = np.nan
dfr.loc[mask_pos, sn_bint] = 1.0
dfr.loc[mask_npos, sn_bint] = 0.0
# dfr.loc[mask_npos, sn_bint] = 0.25
dfr.loc[mask_neg, sn_bint] = 0.0
# dfr.loc[mask_nneg, sn_bint] = 0.75
dfr.loc[mask_nneg, sn_bint] = 1.0

columns_final = [hi, up, dn, pm, sn_bint, at_orig, at, ng, from_abstract, prec, sc, 'plo']
dfr = dfr[columns_final]

with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}_full.pgz'.format(origin, version)), 'wb') as fp:
    pickle.dump(dfr, fp)

# only pos/neg
dfa = dfr[dfr[at].notnull()]
print_stats(dfa)

# only pos/neg from abstract
dfb = dfa[dfa[from_abstract]].copy()
print_stats(dfb)

# only pos/neg from abstract with known precision
dfc = dfb[dfb[sc].notnull()].copy()
print_stats(dfc)

dfc[sc + '_shifted'] = dfc[sc] - dfc[sc].min() + 0.01

df_bint_est = dfc.groupby([up, dn, pm],
                          group_keys=False).apply(lambda g:
                                                  np.sum(g[sn_bint]*g[sc + '_shifted'])/g[sc + '_shifted'].sum())
df_bint_est = df_bint_est.reset_index().rename(columns={0: 'bint_est'})

df_score_est = dfc.groupby([up, dn, pm],
                           group_keys=False).apply(lambda g: g[sc].sum())
df_score_est = df_score_est.reset_index().rename(columns={0: 'score_est'})

dfw = df_bint_est.merge(df_score_est, on=[up, dn, pm])

dfw = dfw.merge(dfc[[up, dn, pm, at_orig, from_abstract, 'plo']],
                on=[up, dn, pm], how='left')

# claims pos/neg, from abstracts, with known precision / without features glued
with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}_cut_raw.pgz'.format(origin, version)), 'wb') as fp:
    pickle.dump(dfr, fp)

df_pmid = df_pmid[~df_pmid['issn'].isnull()]

df_pmid['issn_str'] = df_pmid['issn']
df_pmid['issn'] = df_pmid['issn'].apply(issn2int)

# drop pmids without years
df_pmid = df_pmid.loc[~df_pmid['year'].isnull()]

# convert years to int
df_pmid['year'] = df_pmid['year'].astype(int)
set_pmids_issns = set(df_pmid['issn'].unique())

set_ai_issns = set(df_ai['issn'].unique())
print('{0} issns in pmids-issn table that are not ai table'.format(len(set_pmids_issns - set_ai_issns)))
print('{0} issns in pmids-issn table that are ai table'.format(len(set_pmids_issns & set_ai_issns)))
working_pmids = set(dfr['pmid'].unique())
issn_pmids = set(df_pmid['pmid'].unique())
print('{0} of pmids from geneways that are not in pmid-issn table'.format(len(working_pmids - issn_pmids)))
mask = df_pmid['issn'].isin(list(set_ai_issns))
print('{0} of pmids in pmid-issn table that are in issn-ai table'.format(sum(mask)))

# cut (pm-issn) to issns only in (issn-ye-aiai)
df_pmid2 = df_pmid.loc[mask]
dfw2 = pd.merge(dfw, df_pmid2, on=pm)
# merge (pm-issn) aonto (claims)

fraction = float(dfw2.shape[0]) / dfr.shape[0]
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
dfw3 = pd.merge(dfw2, df_feature_cut, on=pm)

df_affs = df_affs.drop_duplicates(pm)

dfw3 = pd.merge(dfw3, df_affs, how='left', on=pm)

dfw3[ar] = dfw3[ar].fillna(-1)

# claims pos/neg, from abstracts, with known precision / with features glued
with gzip.open(expanduser('~/data/kl/claims/df_{0}_{1}.pgz'.format(origin, version)), 'wb') as fp:
    pickle.dump(dfw3, fp)
