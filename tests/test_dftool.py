import datahelpers.dftools as dt
import pandas as pd
import unittest


class TestDFTolls(unittest.TestCase):

    def test_create_unique_index(self):

        df0 = pd.DataFrame({'i1': [0, 1, 2, 2, 2, 3],
                            'i2': [0, 0, 1, 2, 3, 4],
                            'i3': ['a', 'a', 'a', 'a', 'a', 'a']})

        cases = [None, 'first', 'second', 'both']
        answers = [df0.shape[0], df0.shape[0]-1, df0.shape[0]-2, df0.shape[0]-3]
        answers2 = []
        for c in cases:
            dfr = dt.create_unique_index(df0, ['i1', 'i2'], c)
            answers2.append(len(dfr['i1xi2'].unique()))

        self.assertListEqual(answers, answers2)

if __name__ == '__main__':
    unittest.main()