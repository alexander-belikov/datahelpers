from ..collapse import (
    collapse_series_simple,
    collapse_strings,
    convert_NAs_Series,
    convert_NAs_DataFrame,
    aggregate_negatives_boolean_style,
)
from pandas import Series, DataFrame, concat
import numpy as np
from numpy import random as rn
import unittest


class TestCollapse(unittest.TestCase):

    N = 60
    strings = ["a", "aa", "ab", "ac", "aaa", "bbb", "kkk"]
    strings_NAs = ["a", "NULL", "ab", "NAN", "aaa", "bbb", "nan"]

    inds = rn.randint(0, len(strings) - 2, N)
    inds2 = rn.randint(0, len(strings), N)
    data = [strings[j] for j in inds]
    data2 = [strings[j] for j in inds2]
    data3 = np.array([strings_NAs, strings_NAs[::-1]])

    c1, c2 = "c1", "c2"
    index_cols = [c1, c2]

    N2 = 10
    at = "action"
    st = "claim"
    cols_total = [c1, c2, at, st]
    keys = ["a", "b"]
    vals = [[True, True], [True, False], [False, True], [False, False]]
    s = np.array(keys)

    index_c = np.tile(s, (N2, 1))
    tt = np.tile(vals[0], (2, 1))
    ft = np.tile(vals[1], (2, 1))
    tf = np.tile(vals[2], (2, 1))
    ff = np.tile(vals[3], (4, 1))
    data_bool = np.vstack([tt, ft, tf, ff])
    dfa = DataFrame(index_c, columns=index_cols)
    dfb = DataFrame(data_bool, columns=[at, st])
    df = concat([dfa, dfb], axis=1)

    def test_collapse_series_simple(self):
        s = Series(self.data)
        s2 = Series(self.data2)

        ret = collapse_series_simple(s)
        ddinv = ret[1]
        ret3 = collapse_series_simple(s2, ddinv)

        self.assertEquals(set(ret3[1].keys()), set(self.strings))

    def test_collapse_strings(self):
        df = DataFrame(np.reshape(self.data, (-1, 2)), columns=self.index_cols)
        df2 = DataFrame(np.reshape(self.data2, (-1, 2)), columns=self.index_cols)

        ret = collapse_strings(df, working_columns=df.columns)
        ddinv = ret[1]
        ret3 = collapse_strings(df2, str_dicts=ddinv, working_columns=df2.columns)

        self.assertEquals(ret3[1], ret[1])

    def test_convert_NAs_Series(self):
        s = Series(self.strings_NAs)
        s = convert_NAs_Series(s)
        numberNAs = sum(s.isnull())
        self.assertEquals(numberNAs, 3)

    def test_convert_NAs_DataFrame(self):
        df = DataFrame(self.data3.T, columns=self.index_cols)
        df = convert_NAs_DataFrame(df, dropNAs=True, working_columns=self.index_cols)
        self.assertEquals(df.shape[0], 2)

    def test_aggregate_negatives_boolean_style(self):
        aggregate_negatives_boolean_style(self.df, self.index_cols, self.at, self.st)


if __name__ == "__main__":
    unittest.main()
