import unittest

import pandas as pd

from teams import Teams


class TestTeams(unittest.TestCase):

    def test_extract_pitcher_team(self):
        input_ = pd.Series(
            {'topTeam': '中日',
             'bottomTeam': '阪神',
             'isBottom': 0
             }
        )
        expected = Teams.HANSHIN
        output = Teams.extract_pitcher_team(input_)
        self.assertEqual(output, expected)

        input_ = pd.Series(
            {'topTeam': '中日',
             'bottomTeam': '阪神',
             'isBottom': 1
             }
        )
        expected = Teams.CHUNICHI
        output = Teams.extract_pitcher_team(input_)
        self.assertEqual(output, expected)

        input_ = pd.Series(
            {'topTeam': '中日',
             'bottomTeam': 'マリナーズ',
             'isBottom': 0
             }
        )
        with self.assertRaises(ValueError):
            output = Teams.extract_pitcher_team(input_)

        input_ = pd.Series(
            {'topTeam': 'ヤンキース',
             'bottomTeam': '阪神',
             'isBottom': 1
             }
        )
        with self.assertRaises(ValueError):
            output = Teams.extract_pitcher_team(input_)

    def test_extract_batter_team(self):
        input_ = pd.Series(
            {'topTeam': '中日',
             'bottomTeam': '阪神',
             'isBottom': 0
             }
        )
        expected = Teams.CHUNICHI
        output = Teams.extract_batter_team(input_)
        self.assertEqual(output, expected)

        input_ = pd.Series(
            {'topTeam': '中日',
             'bottomTeam': '阪神',
             'isBottom': 1
             }
        )
        expected = Teams.HANSHIN
        output = Teams.extract_batter_team(input_)
        self.assertEqual(output, expected)

        input_ = pd.Series(
            {'topTeam': '中日',
             'bottomTeam': 'マリナーズ',
             'isBottom': 1
             }
        )
        with self.assertRaises(ValueError):
            output = Teams.extract_batter_team(input_)

        input_ = pd.Series(
            {'topTeam': 'ヤンキース',
             'bottomTeam': '阪神',
             'isBottom': 0
             }
        )
        with self.assertRaises(ValueError):
            output = Teams.extract_batter_team(input_)


if __name__ == '__main__':
    unittest.main()
