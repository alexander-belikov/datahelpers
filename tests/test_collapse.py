import datahelpers.collapse as dc
import pandas as pd
import numpy as np
from numpy import random as rn
import unittest


class TestCollapse(unittest.TestCase):

    N = 60
    strings = ['a', 'aa', 'ab', 'ac', 'aaa', 'bbb', 'kkk']

    inds = rn.randint(0, len(strings)-2, N)
    inds2 = rn.randint(0, len(strings), N)
    data = [strings[j] for j in inds]
    data2 = [strings[j] for j in inds2]

    cols = ['c1', 'c2']

    def test_collapse_series_simple(self):
        s = pd.Series(self.data)
        s2 = pd.Series(self.data2)

        ret = dc.collapse_series_simple(s)
        ddinv = ret[1]
        ret3 = dc.collapse_series_simple(s2, ddinv)

        self.assertEquals(set(ret3[1].keys()), set(self.strings))

    def test_collapse_strings(self):
        df = pd.DataFrame(np.reshape(self.data, (-1, 2)), columns=['c1', 'c2'])
        df2 = pd.DataFrame(np.reshape(self.data2, (-1, 2)), columns=['c1', 'c2'])

        ret = dc.collapse_strings(df)
        ddinv = ret[1]
        ret3 = dc.collapse_strings(df2, str_dicts=ddinv)

        self.assertEquals(ret3[1], ret[1])


if __name__ == '__main__':
    unittest.main()
