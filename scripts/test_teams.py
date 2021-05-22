import unittest

import pandas as pd

from teams import Teams


class TestTeams(unittest.TestCase):

    def test_extract_pitcher_team(self):
        input_ = pd.Series({
            'bgTop': 3,
            'bgBottom': 2,
            'isBottom': 0
        })
        expected = input_.bgBottom
        output = Teams.extract_pitcher_team(input_)
        self.assertEqual(output, expected)

        input_ = pd.Series(
            {'bgTop': 3,
             'bgBottom': 2,
             'isBottom': 1
             }
        )
        expected = input_.bgTop
        output = Teams.extract_pitcher_team(input_)
        self.assertEqual(output, expected)

    def test_extract_batter_team(self):
        input_ = pd.Series({
            'bgTop': 6,
            'bgBottom': 7,
            'isBottom': 0
        })
        expected = input_.bgTop
        output = Teams.extract_batter_team(input_)
        self.assertEqual(output, expected)

        input_ = pd.Series({
            'bgTop': 6,
            'bgBottom': 7,
            'isBottom': 1
        })
        expected = input_.bgBottom
        output = Teams.extract_batter_team(input_)
        self.assertEqual(output, expected)


if __name__ == '__main__':
    unittest.main()
