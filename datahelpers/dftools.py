import pandas as pd
import numpy as np


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
    :param y:
    :return:
    """
    dfy = df.drop_duplicates(y)
    unique_x = set(df[x].values)
    dropped_x = unique_x - set(dfy[x].values)
    sni_y = df[df[x].isin(dropped_x)].drop_duplicates(y)[y].unique()
    sni_mask = df[y].isin(sni_y)
    return sni_mask, sni_y


def create_unique_index(df_init, columns, respect_axis=None):
    """
    df has only two columns i1 and i2, there are only unique combinations (i1, i2)
    respect_axis can be None, 'first', 'second' or 'both'
    axis from respect_axis is tested for surjective and non injective and the image index is assigned

    :param df_init:
    :param columns:
    :param respect_axis:
    :return:
    """
    df = df_init[columns].copy()
    c_derived = columns[0] + 'x' + columns[1]
    pairs = []
    if respect_axis == 'first':
        pairs = [columns]
    if respect_axis == 'second':
        columns.reverse()
        pairs = [columns]
    if respect_axis == 'both':
        columns_rev = list(columns)
        columns_rev.reverse()
        pairs = [columns, columns_rev]

    masks = {}
    ids = {}
    for p in pairs:
        masks[p[1]], ids[p[1]] = get_sni_mask(df, *p)

    df[c_derived] = 0
    mask_biject = pd.Series([True]*df.shape[0], df.index)

    for k in masks.keys():
        mask_biject ^= masks[k]
    df.loc[mask_biject, c_derived] = range(0, sum(mask_biject))
    current_index = sum(mask_biject)

    for k in masks.keys():
        m = masks[k]
        index_dict = {k: v for (k, v) in zip(ids[k], range(current_index, current_index + len(ids[k])))}
        df.loc[m, c_derived] = df.loc[m, k]
        df.loc[m, c_derived] = df.loc[m, c_derived].apply(lambda x: index_dict[x])
        current_index += len(ids[k])
    return df_init.merge(df, on=columns, copy=False)