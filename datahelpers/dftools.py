import pandas as pd
import numpy as np
import datahelpers.collapse as dc
from constants import protein_cols, triplet_index_cols, integer_agg_index


def analyze_unique(df, column):
    dfr = df.drop_duplicates(df.columns)
    vc = dfr[column].value_counts()
    non_unique_ids = list(vc.loc[(vc > 1)].index)
    df_nonuniq = df.loc[df[column].isin(non_unique_ids)]
    dfr = dfr.loc[~dfr[column].isin(non_unique_ids)]
    return dfr, df_nonuniq


def get_sni_mask(df, x, y):
    """
    sni = surjective and non injective
    :param df:
    :param x:
    :return:
    """

    sni_mask = df.duplicated(y, keep=False)
    sni_y = df.loc[sni_mask, y].unique()
    sni_x = df.loc[sni_mask, x].unique()
    return sni_mask, sni_y, sni_x


def create_unique_index(df_init, columns, sni_index=[], ambiguous_index=None):
    """
    df_init has two columns = [i1, i2]
    a super-index is derived based on
    surjective and non injective maps (sni)
    i1 and i2 are tested for sni map
    i1 key means will stand for i1->i2 map

    any df is split into bijective part, sni_i1, sni_i2 and an ambiguous parts

    if i1 is in sni_index, then the super-index will only index unique i2
    in the sni_i1 subset of df

    e.g. if ambiguous is i1 then the indexing in the ambiguous part is done by i1
    if ambiguous is None the ambiguous part is just dropped

    :param df_init: DataFrame
    :param columns: is pair [i1, i2] from the columns of df_init
    :param sni_index: a subset of columns
    :param ambiguous_index: i1, i2 or None
    :return:
    """

    df = df_init[columns].copy()
    c_derived = columns[0] + 'x' + columns[1]
    x = columns[0]
    y = columns[1]
    z = 'z'
    pairs = {x: y, y: x}

    masks = {}
    ids = {x: {}, y: {}, z: {}}

    for k in columns:
        masks[k], ids[k][pairs[k]], ids[k][k] = get_sni_mask(df, k, pairs[k])

    for k in columns:
        ids[z][k] = list(set(ids[x][k]) & set(ids[y][k]))

    mask_z = (df[x].isin(ids[z][x])) | (df[y].isin(ids[z][y]))

    for k in masks.keys():
        masks[k] &= ~mask_z

    masks[z] = mask_z

    mask_biject = pd.Series([True]*df.shape[0], df.index)

    for k in masks.keys():
        mask_biject &= ~masks[k]

    df[c_derived] = df[columns[0]]

    df.loc[mask_biject, c_derived] = np.arange(0, np.sum(mask_biject))

    current_index = np.sum(mask_biject)

    for k in columns:
        m = masks[k]

        if k in sni_index:
            sni_ids = df.loc[masks[k], pairs[k]].unique()
            number_uniques = len(sni_ids)
            index_dict = {key: v
                          for (key, v) in zip(sni_ids,
                                              np.arange(current_index, current_index + number_uniques))}
            df.loc[m, c_derived] = df.loc[m, pairs[k]].apply(lambda arg: index_dict[arg])
        else:
            number_uniques = np.sum(m)
            df.loc[m, c_derived] = np.arange(current_index, current_index + number_uniques)

        current_index += number_uniques

    if ambiguous_index and ambiguous_index in columns:
        sni_ids = df.loc[mask_z, ambiguous_index].unique()
        number_uniques = len(sni_ids)
        index_dict = {key: v for (key, v) in zip(sni_ids, np.arange(current_index, current_index + number_uniques))}
        df.loc[mask_z, c_derived] = df.loc[mask_z, ambiguous_index].apply(lambda arg: index_dict[arg])

    else:
        df = df.loc[~mask_z]

    df[c_derived] = df[c_derived].astype(np.int64)

    # dfr = df_init.merge(df, on=columns, how='left', copy=False)
    dfr = df_init.merge(df, on=columns, copy=False)

    return dfr


def get_multiplet_to_int_index(df, index_cols=triplet_index_cols, int_index_name='it'):
    """

    :param df:
    :param index_cols:
    :param int_index_name:
    :return:
    """
    # get df with unique t = (t1, t2, ..  ti)
    df2 = df[index_cols].drop_duplicates(index_cols)
    # create proxy integer multiplet integer it <-> t
    df3 = df2.set_index(index_cols).reset_index()
    df4 = df3.reset_index().rename(columns={'index': int_index_name})
    # merge back i_t into df with unique t = (t1, t2, t3) and i_h
    df5 = pd.merge(df, df4, on=index_cols, how='left')

    return df5


def process_df_index(dft0, index_cols=triplet_index_cols, int_index_name=integer_agg_index,
                     prefer_triplet=False):
    """
    the default behaviour is to throw away
    :param dft0:
    :param index_cols:
    :param int_index_name:
    :param prefer_triplet
    :return:
    """

    # get df with unique t = (t1, t2, t3) and i_h
    dfw = dft0[index_cols + ['hiid']].drop_duplicates(index_cols + ['hiid'])
    dfw5 = get_multiplet_to_int_index(dfw, index_cols)
    # create a unique index across i_t and i_h
    # in case there is an ambiguity - prefer 'hiid'
    if prefer_triplet:
        ll = ['hiid']
    else:
        ll = [int_index_name]
    dfw6 = create_unique_index(dfw5, ['hiid', int_index_name], ll)
    df2 = pd.merge(dft0, dfw6, on=index_cols + ['hiid'], how='inner')
    return df2


def regexp_collapse_protein_cols(dft0, collapse_df=True, regexp_columns=protein_cols):
    """

    :param dft0: initial DataFrame
    :param collapse_df: flag - transform object (str) to int
    :param regexp_columns: columns to reduce using regular expressions
    :return:
    """

    df = dft0.copy()
    df, dd, dd_regexp = dc.regexp_reduce_yield_agg_dict(df, regexp_columns)
    df_dd = {}

    if collapse_df:
        for c in protein_cols:
            df_dd[c] = dd
        df, dd_dd = dc.collapse_df(df, str_dicts=df_dd, working_columns=regexp_columns)
    return df, df_dd


def compute_centralities(df, node_cols, edge_type, extra_attr):
    """

    :param df:
    :param node_cols:
    :param edge_type:
    :param extra_attr:
    :return:
    """
    dict_keys = {'uni_nodes': node_cols,
                 'uni_nodes_edge': node_cols+edge_type,
                 'uni_nodes_edge_extra': node_cols+edge_type+extra_attr}

    # uni_nodes - number of connections a -> b for a give 'a' over diff 'b'
    # uni_nodes_edge - number of connection a->c->b, where c is the type of edge
    # for a give 'a' over different 'b' and 'c'
    # uni_nodes_edge - number of connection a->c->b (e), where c is the type of edge
    # for a give 'a' over different 'b' and 'c' and 'e'

    dict_dfs = {}
    for k in dict_keys.keys():
        dict_dfs[k] = df[dict_keys[k]].drop_duplicates(dict_keys[k])

    for d in node_cols:
        for k in dict_dfs.keys():
            dft = dict_dfs[k].groupby(d).apply(lambda x: x.shape[0]).sort_values()
            dft = dft.rename('centr_' + d + '_' + k)
            df = pd.merge(df, pd.DataFrame(dft), how='left', left_on=d, right_index=True)
    return df


def XOR(s1, s2):
    return ~(s1 | s2) | (s1 & s2)
