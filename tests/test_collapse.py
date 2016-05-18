import datahelpers.collapse as dc
import pandas as pd
import numpy as np
from numpy import random as rn
import unittest


class TestCollapse(unittest.TestCase):

    N = 60
    strings = ['a', 'aa', 'ab', 'ac', 'aaa', 'bbb', 'kkk']
    strings_NAs = ['a', 'NULL', 'ab', 'NAN', 'aaa', 'bbb', 'nan']

    inds = rn.randint(0, len(strings)-2, N)
    inds2 = rn.randint(0, len(strings), N)
    data = [strings[j] for j in inds]
    data2 = [strings[j] for j in inds2]
    data3 = np.array([strings_NAs, strings_NAs[::-1]])

    cols = ['c1', 'c2']

    def test_collapse_series_simple(self):
        s = pd.Series(self.data)
        s2 = pd.Series(self.data2)

        ret = dc.collapse_series_simple(s)
        ddinv = ret[1]
        ret3 = dc.collapse_series_simple(s2, ddinv)

        self.assertEquals(set(ret3[1].keys()), set(self.strings))

    def test_collapse_strings(self):
        df = pd.DataFrame(np.reshape(self.data, (-1, 2)), columns=self.cols)
        df2 = pd.DataFrame(np.reshape(self.data2, (-1, 2)), columns=self.cols)

        ret = dc.collapse_strings(df)
        ddinv = ret[1]
        ret3 = dc.collapse_strings(df2, str_dicts=ddinv)

        self.assertEquals(ret3[1], ret[1])

    def test_convert_NAs_Series(self):
        s = pd.Series(self.strings_NAs)
        s = dc.convert_NAs_Series(s)
        numberNAs = sum(s.isnull())
        self.assertEquals(numberNAs, 3)

    def test_convert_NAs_DataFrame(self):
        df = pd.DataFrame(self.data3.T, columns=self.cols)
        df = dc.convert_NAs_DataFrame(df, dropNAs=True, working_columns=self.cols)
        self.assertEquals(df.shape[0], 2)

if __name__ == '__main__':
    unittest.main()
