import numpy as np
import pandas as pd
import datahelpers.collapse as dc
import datahelpers.dftools as dfto
from wos_parser.parse import issn2int
import pickle
import gzip
import json
from os.path import expanduser

#TODO revamp min year for ef_ai

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
        while right-left > 1:
            mid = (right+left) // 2
            if (x - years[left])*(years[mid] - x) > 0:
                right = mid
            else:
                left = mid
        return years[left]


def drop_duplicates_cols_arrange_colA(df, cols, colA):
    # drop (ni, pm) duplicates while leaving only max prec values out degenerate
    # lucky for us
    # set 'prec' to float and assign mean value to null
    print(df.shape)
    mask = (df[colA] == 'NULL')
    df.loc[~mask, colA] = df.loc[~mask, colA].astype(float)
    idx = (df.loc[~mask].groupby(cols)[colA].transform(max) == df.loc[~mask, colA])
    print(sum(idx))
    mean_colA = df.loc[~mask, colA].loc[idx].mean()
    df.loc[mask, colA] = mean_colA
    df = df.drop_duplicates(cols)
    print(df.shape, df[colA].dtype)
    return df


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
ai = 'ArticleInfluence'
ni = 'new_index'

dfdd = {}
dfi2, dfdd = dc.collapse_df(dfi, str_dicts=dfdd, dropna_columns=[pm, hi],
                            bool_columns=[ng, ft],
                            numeric_columns=[hi, sn, am_id, pm, sc, pr])

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
print('n ids in pmids-issn table {0} but not ai table'.format(len(set_pmids_issns - set_ai_issns)))
print('n ids in pmids-issn table {0} and in ai table'.format(len(set_pmids_issns & set_ai_issns)))
working_pmids = set(dfi2['pmid'].unique())
issn_pmids = set(df_pmid['pmid'].unique())
print('number of pmids from geneways that are not in pmid-issn table', len(working_pmids - issn_pmids))
mask = df_pmid['issn'].isin(list(set_ai_issns))
print('number of pmids in pmid-issn table that are in issn-ai table'.format(sum(mask)))


# cut (pm-issn) to issns only in (issn-ye-aiai)
df_pmid2 = df_pmid.loc[mask]
dfi3 = pd.merge(dfi2, df_pmid2, on='pmid')
# merge (pm-issn) onto (claims)
print(dfi2.shape[0], dfi3.shape[0], '{0} fraction of claims with identifiable issn '.format(dfi3.shape[0]/dfi2.shape[0]))
### conclusion : 85 % of issns are covered


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
df_ai = df_ai.rename(columns={'year':'ai_year'})
df_feature = pd.merge(df_pmid3, df_ai, left_on=['issn', 'proxy_year'], right_on=['issn', 'ai_year'])
df_feature_cut = df_feature[['pmid', 'ai_cdf']]
dfi4 = pd.merge(dfi3, df_feature_cut, on=pm)

# merge (id-pm-claims-issn-ye-ai) onto (id-up-dn)

df_ha = pd.read_csv('~/data/kl/raw/human_action.txt.gz',
                    sep='\t', index_col=None, compression='gzip')

with open(expanduser('~/data/kl/claims/actions_v2.json'), 'r') as fp:
    agg_act_dict = json.load(fp)

bools = {'true': True, 'false': False}

agg_act_dict = {bools[k]: v for k, v in agg_act_dict.items()}

invdd = dc.invert_dict_of_list(agg_act_dict)

df_ni = dfto.attach_new_index(df_ha, invdd, [at_orig, at], [up2, dn2], ni)
dfi5 = dfi4.merge(df_ni[[hi, ni, up2, dn2, at]], on=hi)
print(dfi4.shape, dfi5.shape)
dfi5 = drop_duplicates_cols_arrange_colA(dfi5, [ni, pm], pr)
print(dfi5.shape)
print(sum(dfi5['year'] < 1990), dfi5.shape[0], 
      '{0} fraction of pmids before 1990'.format(sum(dfi5['year'] < 1990)/dfi5.shape[0]))


# create positive claim column
# new_index is defined up, dn pair; 
# the actions can as positive and negative
# by default the action is deemed positive
# the claim
mask_negs = ~(dfi5[at])
print(sum(mask_negs), mask_negs.shape[0])
dfi5[ps] = ~dfi5[ng]
dfi5.loc[mask_negs, ps] = dfi5.loc[mask_negs, ng]
print('number of negative claims {0}'.format(sum(dfi5[ng])))
print('number of negaive claims on reduced statements {0}'.format(sum(~dfi5[ps])))
dfi5[ps] = dfi5[ps].astype(int)
dfi5[at] = dfi5[at].astype(int)

with gzip.open(expanduser('~/data/kl/claims/df_cs_9.pgz'), 'wb') as fp:
    pickle.dump(dfi5, fp)
