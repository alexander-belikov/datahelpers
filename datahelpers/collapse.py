import numpy as np
from pandas import to_numeric, DataFrame


def convert_to_bool(df, inplace=False):
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

    for c in dft.columns:
        if str('NULL') in dft[c].unique():
            dft[c] = dft[c].replace({'NULL': np.nan})
        if len(dft[c].unique()) < 3:
            # perhaps something more sophisticated could be
            # implemented
            dft[c] = dft[c].replace({'Y': True, 'N': False,
                                     '1': True, '0': False})
    return dft


def convert_to_numeric(df, inplace=False):
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
    for c in dft.columns:
        dft[c] = to_numeric(dft[c], errors='ignore')
        try:
            s = dft[c].astype(int)
            if s == dft[c]:
                dft[c] = s
        except:
            pass
    return dft


def collapse_df(df, str_dicts=None):
    """
    collapses DataFrame types column by column
    :param df: DataFrame
        df to transform
    :param str_dicts: dict
        dictionary of string conversion dictionaries corresponding to columns of df
    :return:
    """

    df = convert_to_bool(df)
    df = convert_to_numeric(df)
    df, dds = collapse_strings(df, str_dicts)
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
    dft['mask'] = (s2 != s2.shift())
    narr = dft['mask'].values
    narr2 = np.ndarray(shape=(narr.shape[0]), dtype=int)
    it = 0
    narr[0] = 1
    for k in range(narr.shape[0]):
        if narr[k] and k > 0:
            it += 1
        narr2[k] = it
    dft['coded'] = narr2
    dft.sort_index(inplace=True)
    return dft['coded'], dd


def collapse_series_simple(s, ddinv=None):
    """
    encode Series s of objects into a Series of ints and provide the encoding dict
        dd format
        dd = {obj1 : int1, obj2 : int2}
        ddinv format
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
            int_max = max(ddinv.keys())+1
            dd_extra = {(k+int_max): list_extra[k] for k in
                        np.arange(len(list_extra))}
            ddinv.update(dd_extra)
        dd = {ddinv[k]: k for k in ddinv.keys()}

    else:
        dd = {ll[k]: k for k in np.arange(ll.shape[0])}
        ddinv = {dd[k]: k for k in dd.keys()}
    s = s.apply(lambda x: dd[x])
    return s, ddinv

collapse = collapse_series_simple


def collapse_strings(df_orig, n=None, str_dicts=None, verbose=False):
    """
    encode DataFrame's constituent Series of objects into a Series of ints
    and provide the encoding dict of dicts

    :param df_orig:
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
    if isinstance(df, DataFrame):
        for c in df.columns:
            if df[c].dtype == np.dtype('O'):
                if verbose:
                    print 'process column', c
                if c not in str_dicts.keys():
                    str_dicts[c] = None
                df[c], str_dicts[c] = collapse(df[c], str_dicts[c])
    else:
        df, str_dicts = collapse(df)
    return df, str_dicts

# collapse simple seems to win
# %time dft_sort, dd_sort = collapse_strings(df)
# %time dft, dd = collapse_strings(df, sort_method=False)
# CPU times: user 2.74 s, sys: 79.8 ms, total: 2.82 s
# Wall time: 2.81 s
# CPU times: user 988 ms, sys: 32.8 ms, total: 1.02 s
# Wall time: 1.02 s
