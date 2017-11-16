import pandas as pd
from bm_support import disambiguation as di
import re
import multiprocessing as mp
from os.path import expanduser

aff = 'affiliation'
pm = 'pmid'
aff_id = 'aff_id'
rank_id = 'rank_id'
rating = 'rating'
inst = 'Institution'
rank = 'Rank'
aca = 'Academic'
cit = 'Citations'
sco = 'Score'
cit2 = 'cite'
targets = ['instit', 'cent', 'college', 'ecole', 'universit', 'school', 'hospital', 'laborat']
# number of proccesses too
n_pieces = 3


def eat_cali(s):
    """
    cut out a dash between alpha and numeric
        example: 'oct-434' -> 'oct434'
    """
    return re.sub(r'(?<=of California)+(,)', '', s)


def prepare_pmid_affs():
    df = pd.read_csv(expanduser('~/data/kl/affs/integral_0_medline_affs.csv.gz'),
                     sep='|', compression='gzip',
                     index_col=False)
    df = df[df.columns[1:]]
    df_affs_ids = df[aff].drop_duplicates()
    df_affs_ids = df_affs_ids.reset_index().rename(columns={'index': aff_id})[[aff_id, aff]]
    # eat cali
    df_affs_ids[aff] = df_affs_ids[aff].apply(lambda x: eat_cali(x))
    df_pmid_aff_id = pd.merge(df[[pm, aff]], df_affs_ids, on=aff)[[pm, aff_id]]
    print(df.shape, df_affs_ids.shape)
    return df_pmid_aff_id, df_affs_ids


def prepare_ratings(n_rating=5):
    dfr = pd.read_csv(expanduser('~/data/rankings/QS WORLD UNIVERSITY RANKINGS 2011 - Biological Sciences.csv'))
    cols = [inst, cit, sco]
    dfr2 = dfr[cols].rename(columns={inst: aff, cit: cit2})

    # eat cali
    dfr2[aff] = dfr2[aff].apply(lambda x: eat_cali(x))

    # we choose citation ranking
    dfr3 = dfr2[[aff, cit2]].sort_values(cit2, ascending=False).reset_index(drop=True)
    dfr3 = dfr3.reset_index()
    delta = dfr3.shape[0] // n_rating
    dfr3[rating] = dfr3['index'].apply(lambda x: n_rating - x // delta)
    dfr3 = dfr3.rename(columns={'index': rank_id})
    return dfr3[[rank_id, aff, rating]]


df_pms, dfa = prepare_pmid_affs()

dfb = prepare_ratings()

dfs = di.split_df(dfa.head(20), n_pieces)

barebone_dict_pars = {'dfb': dfb, 'full_report': False,
                      'targets': targets}

kwargs_list = [{**barebone_dict_pars, **di.generate_fnames(j),
                **{'dfa': df}} for j, df in
               zip(range(n_pieces), dfs)]

qu = mp.Queue()

processes = [mp.Process(target=di.wrapper_disabmi, kwargs=kw)
             for kw in kwargs_list]

for p in processes:
    p.start()

for p in processes:
    p.join()

fnames = [di.generate_fnames(j) for j in range(n_pieces)]
dfs_agg = [pd.read_csv(fn['fname'], compression='gzip', index_col=0) for fn in fnames]
aff_aff_map = pd.concat(dfs_agg)

# aff_aff_map[aff_id, rank_id(ranking)], dfb [rank_id, rating] , df_pms [pmid, aff_id]
# result : [pmid, rating]
print(df_pms.shape)
m2 = pd.merge(aff_aff_map, dfb[[rank_id, rating]], on=rank_id, how='left')
m3 = pd.merge(df_pms, m2, on=aff_id, how='left')
m3[rating] = m3[rating].fillna(0)
m3.head()
m3[[pm, rating]].to_csv(expanduser('~/tmp/pmid_ranking.csv.gz'), compression='gzip')
