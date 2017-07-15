import pandas as pd
import bm_support.gene_id_converter as bgc
import datahelpers.dftools as dfto
from os.path import expanduser
from itertools import product
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

up = 'up'
dn = 'dn'
ps = 'pos'
at = 'action'
at = 'pos'
ni = 'new_index'
pm = 'pmid'
ye = 'year'

df = pd.read_csv(expanduser('~/data/literome/pathway-extraction.txt.gz'), sep='\t', compression='gzip')

# to save pmids so that years and issn could be pulled later
# df['PMID'].to_csv('/Users/belikov/data/literome/literome_pmids.csv.gz', compression='gzip', index=None)
pmids = df['PMID'].drop_duplicates()
# pmids2 = pd.DataFrame(pmids.unique(), columns=[pm])
pmids2 = pd.DataFrame(pmids.values, index=pmids.index, columns=[pm]).reset_index()

mcut = (df['Theme'].str.contains('_')) | (df['Cause'].str.contains('_'))
print(float(sum(~mcut))/mcut.shape[0])

df = df[~mcut]

df[up] = df['Cause'].apply(lambda x: x.split(':')[-1])
df[dn] = df['Theme'].apply(lambda x: x.split(':')[-1])
m = (df['Regulation Type'] == 'Positive_regulation')

df[at] = True
df.loc[~m, at] = False

cols = [up, dn, at]

df_tmp = df[cols]

p, q = unfold_df(df_tmp)

dfs = pd.DataFrame(q, columns=(['index'] + list(p)))

gc = bgc.GeneIdConverter(expanduser('~/data/chebi/hgnc_complete_set.json.gz'), bgc.types, bgc.enforce_ints)
gc.choose_converter('symbol', 'entrez_id')

set_symbols = gc.convs['symbol', 'entrez_id']
m_up = dfs[up].isin(set_symbols)
m_dn = dfs[dn].isin(set_symbols)
dfr = dfs[m_up & m_dn].copy()

gc.choose_converter('symbol', 'entrez_id')
dfr[up] = dfr[up].apply(lambda x: gc[x])
dfr[dn] = dfr[dn].apply(lambda x: gc[x])

dfr2 = dfto.get_multiplet_to_int_index(dfr, [up, dn], ni)


with gzip.open(expanduser('~/data/kl/raw/medline_doc_cs_4.pgz'), 'rb') as fp:
    df_pmid = pickle.load(fp)


pmids3 = pd.merge(pmids2, df_pmid, how='inner', on=pm)


dfr3 = pd.merge(dfr2, pmids3, on='index', how='left')


# df_pmid = df_pmid[~df_pmid['issn'].isnull()]

# df_pmid['issn_str'] = df_pmid['issn']
# df_pmid['issn'] = df_pmid['issn'].apply(issn2int)

# drop pmids without years
dfr4 = dfr3.loc[~dfr3[ye].isnull()].copy()
dfr4 = dfr4.loc[~dfr4[pm].isnull()].copy()
dfr4[pm] = dfr4[pm].astype(int)
dfr4[ye] = dfr4[ye].astype(int)
dfr4[at] = dfr4[at].astype(int)
dfr5 = dfr4[[ni, up, dn, at, ye]]
# convert years to int
# df_pmid['year'] = df_pmid['year'].astype(int)
# set_pmids_issns = set(df_pmid['issn'].unique())

k = pm
m = dfr4[k].isnull()
print('number of mising {0} {1}, while total number of rows is {2}'.format(k, sum(m), dfr4.shape[0]))


with gzip.open(expanduser('~/data/kl/claims/df_lit_2.pgz'), 'wb') as fp:
    pickle.dump(dfr5, fp)
