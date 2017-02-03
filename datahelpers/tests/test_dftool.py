from pandas import DataFrame
import datahelpers.dftools as dt
from ..dftools import create_unique_index
from unittest import TestCase, main


class TestDFTolls(TestCase):
    df0 = DataFrame({'i1': [0, 1, 2, 2, 2, 3, 4, 5, 5],
                     'i2': [0, 0, 1, 2, 3, 4, 5, 5, 6],
                     'i3': ['a']*9})

    not_sni = [1, 2, 3, 4, 5, 0]
    i1_sni = [1, 1, 2, 3, 4, 0]
    i2_sni = [1, 2, 3, 3, 3, 0]
    i1_sni_ambi_i1 = [1, 1, 2, 3, 4, 0, 5, 6, 6]
    i1_sni_ambi_i2 = [1, 1, 2, 3, 4, 0, 5, 5, 6]
    i2_sni_ambi_i1 = [1, 2, 3, 3, 3, 0, 4, 5, 5]
    i2_sni_ambi_i2 = [1, 2, 3, 3, 3, 0, 4, 4, 5]

    def test_create_unique_index_not_sni(self):

        dfr = create_unique_index(self.df0, ['i1', 'i2'])
        self.assertListEqual(list(dfr['i1xi2'].values), self.not_sni)

    def test_create_unique_index_i1_sni(self):

        dfr = dt.create_unique_index(self.df0, ['i1', 'i2'], ['i1'])
        self.assertListEqual(list(dfr['i1xi2'].values), self.i1_sni)

    def test_create_unique_index_i1_sni_ambi_i1(self):

        dfr = create_unique_index(self.df0, ['i1', 'i2'], ['i1'], 'i1')
        self.assertListEqual(list(dfr['i1xi2'].values), self.i1_sni_ambi_i1)

    def test_create_unique_index_i1_sni_ambi_i2(self):

        dfr = create_unique_index(self.df0, ['i1', 'i2'], ['i1'], 'i2')
        self.assertListEqual(list(dfr['i1xi2'].values), self.i1_sni_ambi_i2)

    def test_create_unique_index_i2_sni(self):

        dfr = create_unique_index(self.df0, ['i1', 'i2'], ['i2'])
        self.assertListEqual(list(dfr['i1xi2'].values), self.i2_sni)

    def test_create_unique_index_i2_sni_ambi_i1(self):

        dfr = create_unique_index(self.df0, ['i1', 'i2'], ['i2'], 'i1')
        self.assertListEqual(list(dfr['i1xi2'].values), self.i2_sni_ambi_i1)

    def test_create_unique_index_i2_sni_ambi_i2(self):

        dfr = create_unique_index(self.df0, ['i1', 'i2'], ['i2'], 'i2')
        self.assertListEqual(list(dfr['i1xi2'].values), self.i2_sni_ambi_i2)

if __name__ == '__main__':
    main()
