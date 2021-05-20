import unittest

import numpy as np
import pandas as pd

from players import Players, Hand


class TestAssignPlayerID(unittest.TestCase):

    def test_assign_id(self):
        '''A is duplicated in pitchers
        C appears total 2nd times (1 in pitchers, another in batters)
        F appears 2nd times in batters but they are not same player'''
        pitchers = pd.DataFrame({
            'pitcher': ['A', 'A', 'B', 'C'],
            'pitcherTeam': [1, 1, 2, 3]
        })
        batters = pd.DataFrame({
            'batter': ['C', 'D', 'E', 'F', 'F'],
            'batterTeam': [3, 3, 4, 4, 5]
        })
        players = Players(pitchers=pitchers, batters=batters)
        output = players.assign_id()
        # playerID, player, team
        self.assertTrue('playerID' in output.columns)
        self.assertTrue('player' in output.columns)
        self.assertTrue('team' in output.columns)
        self.assertEqual(output.shape[1], 3)
        # There are 7 players (A, B, C, D, E, F(Team4), F(Team5))
        self.assertEqual(output.shape[0], 7)  # 1 row per 1 player
        self.assertEqual(output.playerID.nunique(), 7)  # 1 ID per 1 player

    def test_assign_1_id_for_transffered_players(self):
        '''3 players added.
        Following 3 players should have only 1 ID across different teams
        'ＤＪ．ジョンソン', '澤村 拓一', '小林 慶祐'
        '''
        pitchers = pd.DataFrame({
            'pitcher': ['A', 'A', 'B', 'C', 'ＤＪ．ジョンソン', 'ＤＪ．ジョンソン', '小林 慶祐'],
            'pitcherTeam': [1, 1, 2, 3, 4, 5, 6]
        })
        batters = pd.DataFrame({
            'batter': ['C', 'D', 'E', 'F', 'F', '澤村 拓一', '澤村 拓一', '小林 慶祐'],
            'batterTeam': [3, 3, 4, 4, 5, 7, 8, 12]
        })
        players = Players(pitchers=pitchers, batters=batters)
        output = players.assign_id()
        # There are 10 players (A, B, C, D, E, F(Team4), F(Team5), 'ＤＪ．ジョンソン', '澤村 拓一', '小林 慶祐')
        self.assertEqual(output.shape[0], 13)  # Last 3 players have 2 rows, others(7) 1 row
        self.assertEqual(output.playerID.nunique(), 10)  # 10 players
        for p in Players.TRANSFERRED_PLAYERS:
            rows = output.query(f'player == "{p}"')
            self.assertEqual(rows.shape[0], 2)  # 2 rows
            self.assertEqual(rows.playerID.nunique(), 1)  # only 1 ID


class TestHand(unittest.TestCase):

    def test_impute_pitcher_hand(self):
        """テストデータ：
        1. 右投げ、欠損は0埋
        2. 左投げ、欠損は1埋
        3. 右投げ、0
        4. 左投げ、0
        """
        input_ = pd.DataFrame({
            'pitcherID': [
                1, 1, 1,
                2, 2, 2,
                3, 3, 3,
                4, 4, 4
            ],
            'pitcherHand': [
                'R',    np.nan, np.nan,
                np.nan, 'L',    'L',
                'R',    'R',    'R',
                'L',    'L',    'L',
            ]
        })
        expected = pd.Series([
            0, 0, 0,
            1, 1, 1,
            0, 0, 0,
            1, 1, 1,
        ])
        output = Hand.is_pitcher_hand_left(input_)
        self.assertTrue(pd.testing.assert_series_equal(output, expected, check_names=False) is None)

    def test_impute_batter_hand(self):
        """テストデータ：
        1. 右打者、0
        2. 左打者、1
        3. 両打者、ピッチャーの反対の手（0→1, 1→0）
        4. 右打者、0
        5. 左打者、1
        6. 両打者、ピッチャーの反対の手（0→1, 1→0）
        """
        input_ = pd.DataFrame({
            'batterID': [
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3,
                4, 4, 4, 4,
                5, 5, 5, 5,
                6, 6, 6, 6,
            ],
            'isPitcherHandLeft': [
                0, 1, 0, 1,
                0, 1, 0, 1,
                0, 1, 0, 1,
                0, 1, 0, 1,
                0, 1, 0, 1,
                0, 1, 0, 1,
            ],
            'batterHand': [
                'R', np.nan, 'R', np.nan,
                'L', np.nan, 'L', 'L',
                'L', 'R', np.nan, 'R',
                'R', 'R', 'R', 'R',
                'L', 'L', 'L', 'L',
                'L', 'R', 'L', 'R',
            ]
        })
        expected = pd.Series([
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 0, 1, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 0, 1, 0,
        ])
        output = Hand.is_batter_hand_left(input_)
        self.assertTrue(pd.testing.assert_series_equal(output, expected, check_names=False) is None)


if __name__ == '__main__':
    unittest.main()
