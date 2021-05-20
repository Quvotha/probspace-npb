import unittest

import pandas as pd

from inning import Inning


class TestInning(unittest.TestCase):

    def test_extract_inning_info(self):
        testdata = [
            ('1回表', pd.Series({'inningNo': 1, 'isBottom': 0})),
            ('1回裏', pd.Series({'inningNo': 1, 'isBottom': 1})),
            ('11回表', pd.Series({'inningNo': 11, 'isBottom': 0})),
            ('11回裏', pd.Series({'inningNo': 11, 'isBottom': 1})),
        ]
        for input_, expected in testdata:
            output = Inning.extract_info(input_)
            self.assertTrue(pd.testing.assert_series_equal(output, expected) is None)

    def test_raise_value_error(self):
        testdata = ['1回', '回表', '回裏', '1表', '一回表', '二回裏']
        for input_ in testdata:
            with self.assertRaises(ValueError):
                Inning.extract_info(input_)


if __name__ == '__main__':
    unittest.main()
