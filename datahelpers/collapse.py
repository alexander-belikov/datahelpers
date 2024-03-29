import numpy as np
from pandas import to_numeric, DataFrame, Series
from .constants import na_names
from .regular import chain_regexp_transforms


def convert_NAs_Series(obj):
    """
    convert various string representations of NaN to np.nan
    :param obj: Series to be
    :return:
    """

    if isinstance(obj, Series):

        na_present = list(set(obj.unique()) & set(na_names))
        # TODO use reduce
        for word in na_present:
            obj = obj.replace(word, np.nan)

        return obj


def convert_NAs_DataFrame(obj, working_columns=[], dropNAs=True):
    """

    :param obj:
    :param inplace:
    :param dropNAs:
    :param working_columns:
    :return:
    """

    cols = list(set(working_columns) & set(obj.columns))
    if isinstance(obj, DataFrame):
        for c in cols:
            # TODO clean inplace redundancy, see pandas
            obj[c] = convert_NAs_Series(obj[c])
        if dropNAs:
            obj = obj.dropna(axis=0, how="any", subset=cols)
        return obj


def convert_to_bool(df, working_columns=[], inplace=False):
    """
    convert all possible columns of DataFrame to bool type
    :param df: DataFrame
        df to transform
    :param inplace: bool
        if True, perform operation in-place
    :return:
        transformed DataFrame
    """
    if inplace:
        dft = df
    else:
        dft = df.copy()

    cols = list(set(working_columns) & set(df.columns))

    for c in cols:
        if len(dft[c].unique()) < 3:
            # perhaps something more sophisticated could be
            # implemented
            dft[c] = dft[c].replace(
                {"Y": True, "N": False, "1": True, "0": False, 1: True, 0: False}
            )
    return dft


def convert_to_numeric(df, working_columns=[], inplace=False):
    """
    convert all possible columns of DataFrame to numeric type
    :param df: DataFrame
        df to transform
    :param inplace: bool
        if True, perform operation in-place
    :return:
        transformed DataFrame
    :return:
    """

    if inplace:
        dft = df
    else:
        dft = df.copy()

    cols = list(set(working_columns) & set(df.columns))

    for c in cols:
        dft[c] = to_numeric(dft[c], errors="ignore")
        try:
            s = dft[c].astype(int)
            if s == dft[c]:
                dft[c] = s
        except:
            pass
    return dft


def collapse_df(
    df,
    str_dicts=None,
    dropna_columns=[],
    bool_columns=[],
    numeric_columns=[],
    object_columns=[],
):
    """
    collapses DataFrame types column by column
    :param df: DataFrame
        df to transform
    :param str_dicts: dict
        dictionary of string conversion dictionaries corresponding to columns of df
    :param working_columns
    :param omit_columns
    :param dropna_columns
    :return:
    """
    # TODO insert datetime clause
    df = convert_NAs_DataFrame(df, dropna_columns)
    df = convert_to_bool(df, bool_columns)
    df = convert_to_numeric(df, numeric_columns)
    df, dds = collapse_strings(df, str_dicts, object_columns)
    return df, dds


def collapse_series_sort(s):
    """
    obsolete
    :param s:
    :return:
    """
    s2 = s.sort_values()
    univals = s2.unique()
    dd = {k: univals[k] for k in np.arange(univals.shape[0])}
    dft = DataFrame(s2)
    dft["mask"] = s2 != s2.shift()
    narr = dft["mask"].values
    narr2 = np.ndarray(shape=(narr.shape[0]), dtype=int)
    it = 0
    narr[0] = 1
    for k in range(narr.shape[0]):
        if narr[k] and k > 0:
            it += 1
        narr2[k] = it
    dft["coded"] = narr2
    dft.sort_index(inplace=True)
    return dft["coded"], dd


def collapse_series_simple2(s, ddinv=None, apply_on_series=True):
    """
    encode Series s of objects into a Series of ints and provide the encoding dict
        dd format
        dd = {obj1 : int1, obj2 : int2}
        ddinv format
    :param apply_on_series:
    :param s:
    :param ddinv: dict
        dict of inverse mapping
        ddinv = {int1 : obj1, int2 : obj2}
    :return:
    """
    ll = np.sort(s.unique())
    if ddinv:
        list_extra = list(set(ll) - set(ddinv.values()))
        if list_extra:
            int_max = max(ddinv.keys()) + 1
            dd_extra = {
                (k + int_max): list_extra[k] for k in np.arange(len(list_extra))
            }
            ddinv.update(dd_extra)
        dd = {ddinv[k]: k for k in ddinv.keys()}

    else:
        dd = {ll[k]: k for k in np.arange(ll.shape[0])}
        ddinv = {dd[k]: k for k in dd.keys()}
    if apply_on_series:
        s = s.apply(lambda x: dd[x])
    return s, ddinv


def collapse_series_simple(s, existing_dict=None, apply_on_series=True):
    """
    :param s: pd.Series
    :param existing_dict: dict
        example: {'a': 1, 'b': 1}
    :param apply_on_series: boolean
    :return: series, dict
        - inverse transform is given by
        ddinv = {dd[k]: k for k in dd.keys()}

    """

    ll = np.sort(s.unique())

    dd = create_renaming_dict(ll, existing_dict)

    if apply_on_series:
        s = s.apply(lambda x: dd[x])
    return s, dd


def create_renaming_dict(obj_list, existing_dict=None):
    """

    :param obj_list:
    :param existing_dict:
    :return:
    """
    if existing_dict:
        list_extra = sorted(list(set(obj_list) - set(existing_dict.keys())))
        if list_extra:
            int_max = max(existing_dict.values()) + 1
            dd = {list_extra[k]: (k + int_max) for k in np.arange(len(list_extra))}
            dd.update(existing_dict)
        else:
            dd = existing_dict
    else:
        dd = {obj_list[k]: k for k in np.arange(len(obj_list))}
    return dd


collapse = collapse_series_simple


def collapse_strings(
    df_orig, str_dicts=None, working_columns=[], n=None, verbose=False
):
    """
    encode DataFrame's constituent Series of objects into a Series of ints
    and provide the encoding dict of dicts

    :param df_orig:
    :param omit_columns:
    :param n: integer
        take n top rows of df_orig
    :param str_dicts: dict of dicts
        str_dicts are concatenated if not None
    :param verbose: bool
        print columns
    :return:
    """

    df = df_orig.head(n).copy()
    if not str_dicts:
        str_dicts = {}

    cols = list(set(working_columns) & set(df.columns))

    if isinstance(df, DataFrame):
        for c in cols:
            if df[c].dtype == np.dtype("O"):
                try:
                    df[c] = df[c].astype(str)
                except UnicodeEncodeError:
                    df[c] = df[c].astype("unicode")
                if verbose:
                    print("process column", c)
                if c not in str_dicts.keys():
                    str_dicts[c] = None
                df[c], str_dicts[c] = collapse(df[c], str_dicts[c])
    else:
        df, str_dicts = collapse(df)
    return df, str_dicts


def regexp_reduce_yield_agg_dict(df, columns):
    """

    :param df:
    :param columns:
    :return:
    """
    agg_encoding_set = set()
    agg_regexp_dict = {}
    for c in columns:
        uni_vals = df[c].unique()
        keys_prime = map(chain_regexp_transforms, uni_vals)
        keys_prime_set = set(keys_prime)
        keys_trans = set(uni_vals) - keys_prime_set
        regexp_dict = {k: chain_regexp_transforms(k) for k in keys_trans}
        # mask of such entries that are regexp transformed
        m = df[c].isin(keys_trans)
        df.loc[m, c] = df.loc[m, c].apply(lambda x: regexp_dict[x])
        agg_encoding_set |= keys_prime_set
        agg_regexp_dict.update(regexp_dict)

    agg_encoding_list = sorted(set(agg_encoding_set))
    agg_encoding_dict = {
        agg_encoding_list[k]: k for k in np.arange(len(agg_encoding_list))
    }
    return df, agg_encoding_dict, agg_regexp_dict


# collapse simple seems to win
# %time dft_sort, dd_sort = collapse_strings(df)
# %time dft, dd = collapse_strings(df, sort_method=False)
# CPU times: user 2.74 s, sys: 79.8 ms, total: 2.82 s
# Wall time: 2.81 s
# CPU times: user 988 ms, sys: 32.8 ms, total: 1.02 s
# Wall time: 1.02 s


def recast_series_input_dict_conversion_dict(s, idict, cdict):
    """

    :param s: Series to recast
    :param idict: input
    :param cdict: conversion dictionary
    :return:
    """

    int_ilist = {cdict[k] for k in idict.keys()}
    s2 = s.copy()
    mask = s.isin(int_ilist)
    uni_vals = list(set(idict.values()))
    new_cdict = {uni_vals[i]: i for i in range(len(uni_vals))}
    mdict = {cdict[k]: new_cdict[idict[k]] for k in idict.keys()}

    s2.loc[mask] = s.loc[mask].apply(lambda x: mdict[x])
    s2.loc[~mask] = np.nan
    return s2, new_cdict


def invert_dict_of_list(dd):
    """
    it is implied that the values in the lists are unique;
    :param dd:
    :return:
    """

    inv_dd = {}
    dts = [{v: k for v in dd[k]} for k in dd.keys()]
    [inv_dd.update(x) for x in dts]
    return inv_dd


def aggregate_negatives_boolean_style(dfi, index_cols, at, st):
    """
    in a DataFrame dfi the statement (L-formula) is identified by index_cols
    it is completed by the column at (action type)
    at is the predicate regarding the statement (True or False)
    it can be interpreted as positive of negative action

    ng is the predicate of the (index_cols, at) statement

    the function finds such pairs (index_cols, at) for all rows in dfi,
    for which at takes two values for each index_cols,
    inverts at and ng for one of the values of at

    ind at  ng

      eff cl
    a   0   0
    a   0   1
    a   1   0
    a   1   1

    :param dfi:
    :param index_cols:
    :param at:
    :param ng:
    :return:
    """
    full_index = index_cols + [at]
    dfw = dfi.reset_index()
    dfw_mod = (
        dfw.groupby(index_cols)
        .apply(lambda x: float(x[at].sum()) / x[at].shape[0] < 0.5)
        .reset_index()
    )
    dfw_mod.rename(columns={0: at}, inplace=True)
    dfw_mod["flag"] = True

    # TODO : publish bug pd.DataFrame.merge does not keep the original index
    mask = dfw[full_index].merge(dfw_mod, how="left", on=full_index)

    mask.loc[mask["flag"].isnull(), "flag"] = False
    m = mask["flag"]
    # print 'number of rows to be transformed', sum(mask_neg_dups_at)
    dfw.loc[m, [at, st]] = ~dfw.loc[m, [at, st]]
    return dfw
