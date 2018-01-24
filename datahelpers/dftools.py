import pandas as pd
import numpy as np
from numpy import concatenate, argsort, cumsum, repeat, unique, vstack
from .constants import protein_cols, triplet_index_cols, integer_agg_index
from .collapse import regexp_reduce_yield_agg_dict

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


def attach_new_index(dfw, redux_dict, map_name, index_cols, new_index_name):
    """

    :param dfw: DataFrame to work on
    :param redux_dict: dict of values' reduction
    :param map_name: tuple of column names (orig, transformed)
    :param index_cols: columns to use as index
    :param new_index_name: new index column name
    :return:
    """
    df_conv = collapse_column(dfw, redux_dict, map_name)
    # create new index based on index_cols, col_reduced is a member of index_cols
    df_conv_new_index = get_multiplet_to_int_index(df_conv, index_cols, new_index_name)

    return df_conv_new_index


def collapse_column(dfw, redux_dict, map_name):
    """

    :param dfw: DataFrame to work on
    :param redux_dict: dict of values reduction
    :param map_name: tuple of column names (orig, transformed)
    :return:
    """
    col_orig, col_reduced = map_name
    # 1) cut the part of dfw, whose domain is the domain of redux_dict map
    mask = dfw[col_orig].isin(redux_dict.keys())

    # 2) create col_reduced in df_conv
    df_conv = dfw.loc[mask].copy()
    df_conv[col_reduced] = df_conv[col_orig].apply(lambda x: redux_dict[x])
    return df_conv


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
    df, dd, dd_regexp = regexp_reduce_yield_agg_dict(df, regexp_columns)
    df_dd = {}

    if collapse_df:
        for c in protein_cols:
            df_dd[c] = dd
        df, dd_dd = collapse_df(df, str_dicts=df_dd, working_columns=regexp_columns)
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


def extract_idc_within_frequency_interval(df, id_col, flag_col, freq_int, min_length=0):
    val_low, val_hi = freq_int
    ps_frac = df.groupby(id_col).apply(lambda x: float(sum(x[flag_col]))/x.shape[0])

    # check - most popular claims
    vc_idt = df[id_col].value_counts()

    vc_idt.name = 'n_claims'
    ps_frac.name = 'ps_frac'
    df_info = pd.merge(pd.DataFrame(vc_idt), pd.DataFrame(ps_frac),
                       left_index=True, right_index=True).sort_values('n_claims', ascending=False)

    m_frac = (df_info['ps_frac'] >= val_low) & \
             (df_info['ps_frac'] <= val_hi) & \
             (df_info['n_claims'] >= min_length)
    dfr = df_info.loc[m_frac].sort_values('n_claims', ascending=False)
    ids = list(dfr.index)
    return ids


def XOR(s1, s2):
    return ~(s1 | s2) | (s1 & s2)


def accumulate_dicts(dict_list):
    """
    convert a list of dicts to one dict
    :param dict_list:
    :return:
    """
    integral_dict = {}
    for item in dict_list:
        integral_dict = {**integral_dict, **item}
    return integral_dict


def dict_to_array(ddict):
    """
        ddict contains arrays of size n \times k_i
        final array has size (n+1) \times \sum k_i
    """
    keys = list(ddict.keys())
    arrays_list = [ddict[k] for k in keys]
    arr = np.concatenate(arrays_list, axis=1)
    keys_list = [[int(k)]*ddict[k].shape[1] for k in keys]
    keys_arr = np.concatenate(keys_list)
    final_array = np.concatenate([keys_arr.reshape(-1, keys_arr.shape[0]), arr])
    return final_array


def select_appropriate_datapoints(df, masks):
    m0 = pd.Series([True]*df.shape[0], df.index)
    for c, thr, foo in masks:
        m = (foo(df[c], thr))
        m0 &= m
    return df.loc[m0].copy()


def drop_duplicates_cols_arrange_col(dft, columns, col):
    # drop rows with col == 'NULL'
    # drop (ni, pm) duplicates
    # only max value of col remain from duplicates
    maskt = (dft[col] == 'NULL')
    print('fraction of claims with missing '
          'precision dropped: {0:.4f}'.format(float(sum(maskt)) / maskt.shape[0]))
    dft2 = dft.loc[~maskt].copy()
    dft2[col] = dft2[col].astype(float)
    dft2 = dft2.reset_index(drop=True)
    idx = dft2.groupby(columns)[col].idxmax()
    dft3 = dft2.loc[idx]
    return dft3


def count_elements_smaller_than_self(x):
    ii = argsort(x)
    ii2 = argsort(ii)
    if ii2.dtype != int:
        print(ii2.dtype, x.shape, ii2[:5])
    uniques, counts = unique(x, return_counts=True)
    csum = [0] + list(cumsum(counts)[:-1])
    cnts = concatenate([repeat(i, c) for i, c in zip(csum, counts)])[ii2]
    return cnts


def count_elements_smaller_than_self_wdensity(x):
    # TODO test for small size x (smaller than 5)
    cnts = count_elements_smaller_than_self(x.values)
    denom = (x.values - np.min(x))
    dns = np.true_divide(cnts, denom, where=(denom!=0))
    r = pd.DataFrame(vstack([cnts, dns]).T, index=x.index)
    return r


def group(seq, sep):
    g = []
    for el in seq:
        if el == sep:
            yield g
            g = []
        else:
            g.append(el)
    yield g


def parse_wos_simple_format(ll, keys):
    info_dict = {}
    # simplified version that works for one line info extraction
    for item in ll:
        morphems = item.split(' ')
        prefix = morphems[0]
        for k in keys:
            if prefix == k:
                info_dict[k] = morphems[1]
    if len(info_dict) == len(keys):
        out = [info_dict[k] for k in keys]
        return out
    else:
        return []


def agg_file_info(fname, keys):
    # parsing web of science plain text dumps
    # webofknowledge.com
    lines = open(fname).read().splitlines()
    lines2 = list(group(lines, ''))
    data = [np.array(parse_wos_simple_format(x, keys)) for x in lines2]
    non_empty_data = list(filter(lambda x: len(x), data))
    y = np.vstack(non_empty_data)
    return y


def add_column_from_file(df, fpath, merge_column, merged_column, impute_mean=True):
    df_cite = pd.read_csv(fpath, compression='gzip', index_col=None)
    print(df_cite.shape)
    df2 = pd.merge(df, df_cite, on=merge_column, how='left')
    df_out = df2.copy()
    if impute_mean:
        mask = df2[merged_column].isnull()
        mean = df2[merged_column].mean()
        df_out.loc[mask, merged_column] = mean
    return df_out
