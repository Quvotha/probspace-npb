import unittest

import numpy as np
import pandas as pd

from players import PlayersID, Hand, GameParticipation


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
        playersid = PlayersID(pitchers=pitchers, batters=batters)
        output = playersid.assign()
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
        playersid = PlayersID(pitchers=pitchers, batters=batters)
        output = playersid.assign()
        # There are 10 players (A, B, C, D, E, F(Team4), F(Team5), 'ＤＪ．ジョンソン', '澤村 拓一', '小林 慶祐')
        self.assertEqual(output.shape[0], 13)  # Last 3 players have 2 rows, others(7) 1 row
        self.assertEqual(output.playerID.nunique(), 10)  # 10 players
        for p in PlayersID.TRANSFERRED_PLAYERS:
            rows = output.query(f'player == "{p}"')
            self.assertEqual(rows.shape[0], 2)  # 2 rows
            self.assertEqual(rows.playerID.nunique(), 1)  # only 1 ID

    def test_compare_2_ids(self):
        '''
        From 1 to 3: Appear both in train/test set.
        From 4 to 6: Appear only in train set.
        From 7 to 9: Appear only int test set.
        '''
        ids_train = pd.Series(
            [
                1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6,
            ]
        )
        ids_test = pd.Series(
            [
                1, 2, 2, 2, 2, 3, 3, 7, 8, 9, 9, 9, 9,
            ]
        )
        expected = (
            [4, 5, 6],  # train only
            [7, 8, 9],  # test only
            [1, 2, 3],  # shared
        )
        output = PlayersID.compare_2_ids(ids_train, ids_test)
        self.assertEqual(output, expected)


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
        self.assertIsNone(pd.testing.assert_series_equal(output, expected, check_names=False))

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
        self.assertIsNone(pd.testing.assert_series_equal(output, expected, check_names=False))


class TestGameParticipant(unittest.TestCase):

    def test_paased_time_from_last_participation(self):
        input_ = pd.DataFrame({
            'gameID': [
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3,
            ],
            'startDayTime': [
                '2020-05-01 10:00:00', '2020-05-01 10:00:00', '2020-05-01 10:00:00', '2020-05-01 10:00:00',
                '2020-05-02 10:00:00', '2020-05-02 10:00:00', '2020-05-02 10:00:00', '2020-05-02 10:00:00',
                '2020-05-04 18:00:00', '2020-05-04 18:00:00', '2020-05-04 18:00:00', '2020-05-04 18:00:00',
            ],
            'pitcherID': [
                1, 2, 3, 5,
                1, 2, 4, 6,
                1, 3, 4, 7,
            ],
            'batterID': [
                10, 20, 30, 50,
                10, 20, 40, 60,
                10, 30, 40, 70,
            ],
        })
        input_['startDayTime'] = pd.to_datetime(input_['startDayTime'])
        game_participation = GameParticipation(data=input_)
        # pitcher
        expected = pd.DataFrame({
            'pitcherID': [
                1, 1, 1,
                2, 2,
                3, 3,
                4, 4,
                5,
                6,
                7,
            ],
            'gameID': [
                1, 2, 3,
                1, 2,
                1, 3,
                2, 3,
                1,
                2,
                3,
            ],
            'hoursElapsed': [
                np.nan, 24, 56,
                np.nan, 24,
                np.nan, 80,
                np.nan, 56,
                np.nan,
                np.nan,
                np.nan,
            ],
            'numGamesParticipated': [
                1, 2, 3,
                1, 2,
                1, 2,
                1, 2,
                1,
                1,
                1,
            ]
        })
        output = game_participation.hours_elapsed_from_last(calc_pitcher=True)
        self.assertIsNone(pd.testing.assert_frame_equal(expected, output))
        # batter
        expected = pd.DataFrame({
            'batterID': [
                10, 10, 10,
                20, 20,
                30, 30,
                40, 40,
                50,
                60,
                70,
            ],
            'gameID': [
                1, 2, 3,
                1, 2,
                1, 3,
                2, 3,
                1,
                2,
                3,
            ],
            'hoursElapsed': [
                np.nan, 24, 56,
                np.nan, 24,
                np.nan, 80,
                np.nan, 56,
                np.nan,
                np.nan,
                np.nan,
            ],
            'numGamesParticipated': [
                1, 2, 3,
                1, 2,
                1, 2,
                1, 2,
                1,
                1,
                1,
            ]
        })
        output = game_participation.hours_elapsed_from_last(calc_pitcher=False)
        self.assertIsNone(pd.testing.assert_frame_equal(expected, output))


if __name__ == '__main__':
    unittest.main()
