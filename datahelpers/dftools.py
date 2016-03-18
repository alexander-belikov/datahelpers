import pandas as pd
import numpy as np
import datahelpers.collapse as dc
from constants import protein_cols, triplet_index_cols


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
    df has only two columns i1 and i2, there are only unique combinations (i1, i2)
    respect_axis can be None, 'first', 'second' or 'both'
    axis from respect_axis is tested for surjective and non injective and the image index is assigned

    :param df_init:
    :param columns:
    :param sni_index:
    :param ambiguous_index:
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

    mask_biject &= ~mask_z

    df[c_derived] = df[columns[0]]

    df.loc[mask_biject, c_derived] = np.arange(0, np.sum(mask_biject))

    current_index = np.sum(mask_biject)

    for k in columns:
        m = masks[k]

        if k in sni_index:
            sni_ids = df.loc[masks[k], pairs[k]].unique()
            number_uniques = len(sni_ids)
            # print k, np.sum(m), sni_ids, number_uniques
            index_dict = {key: v for (key, v) in zip(sni_ids, np.arange(current_index, current_index + number_uniques))}
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

    return df_init.merge(df, on=columns, copy=False)


def process_df_index(dft0, regexp_columns=protein_cols, index_cols=triplet_index_cols, collapse_df=True):
    """

    :param dft0:
    :param regexp_columns:
    :param index_cols:
    :param collapse_df:
    :return:
    """
    df = dft0.copy()
    df, dd, dd_regexp = dc.regexp_reduce_yield_agg_dict(df, regexp_columns)
    df_dd = {}

    if collapse_df:
        for c in protein_cols:
            df_dd[c] = dd
        df, dd2 = dc.collapse_df(df, str_dicts=df_dd, working_columns=regexp_columns)

    # get df with unique t = (t1, t2, t3) and i_h
    dfw = df[index_cols + ['hiid']].drop_duplicates(index_cols + ['hiid'])
    # get df with unique t = (t1, t2, t3)
    dfw2 = dfw[index_cols].drop_duplicates(index_cols)
    # create proxy integer triplet integer i_t <-> t
    dfw3 = dfw2.set_index(index_cols).reset_index()
    dfw4 = dfw3.reset_index().rename(columns={'index': 'i_t'})
    # merge back i_t into df with unique t = (t1, t2, t3) and i_h
    dfw5 = pd.merge(dfw, dfw4, on=index_cols, how='left')
    # create a unique index across i_t and i_h
    dfw6 = create_unique_index(dfw5, ['hiid', 'i_t'], ['i_t'])
    df2 = pd.merge(df, dfw6, on=index_cols + ['hiid'], how='left')
    return df2, dd2


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

    dict_dfs = {}
    for k in dict_keys.keys():
        dict_dfs[k] = df[dict_keys[k]].drop_duplicates(dict_keys[k])

    for d in node_cols:
        for k in dict_dfs.keys():
            dft = dict_dfs[k].groupby(d).apply(lambda x: x.shape[0]).sort_values()
            dft = dft.rename('centr_' + d + '_' + k)
            df = pd.merge(df, pd.DataFrame(dft), how='left', left_on=d, right_index=True)
    return df
