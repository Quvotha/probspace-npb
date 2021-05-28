from functools import partial
from typing import List, Tuple

import numpy as np
import pandas as pd


class PlayersID(object):

    '''
    These players belonged to 2 teams in 2020s.
    https://prob.space/competitions/npb/discussions/Quvotha-nndropout100-Post996ddd42670380ebceb5
    '''
    TRANSFERRED_PLAYERS = ('ＤＪ．ジョンソン', '澤村 拓一', '小林 慶祐')

    def __init__(self, pitchers: pd.DataFrame, batters: pd.DataFrame):
        self.pitchers = pitchers[['pitcher', 'pitcherTeam']]
        self.batters = batters[['batter', 'batterTeam']]

    def assign(self) -> pd.DataFrame:
        """Assign identifier for all players.

        Returns
        -------
        player_ids: pd.DataFrame
            Having 3 columns, `playerID`, `player`, and `team`.
        """
        # Create players list (1 row = 1 player, team)
        player_id = pd.DataFrame({
            'player': pd.concat([self.pitchers.pitcher, self.batters.batter]),
            'team': pd.concat([self.pitchers.pitcherTeam, self.batters.batterTeam])})
        player_id = player_id \
            .drop_duplicates() \
            .sort_values(['team', 'player']) \
            .reset_index(drop=True)
        # Assing ID
        player_id['playerID'] = player_id.index
        for p in self.TRANSFERRED_PLAYERS:
            player_mask = player_id.player == p
            one_id = player_id.loc[player_mask, 'playerID'].min()
            if np.isnan(one_id):
                continue
            else:
                player_id.loc[player_mask, 'playerID'] = one_id
        return player_id

    @staticmethod
    def compare_2_ids(ids_train: pd.Series, ids_test: pd.Series) -> Tuple[List]:
        """Identify which id appears both in train and test set, or dose in only one of them.

        Parameters
        ----------
        ids_trian : pd.Series
            All of the playerIDs which appear in train set.
        ids_test : pd.Series
            All of the playerIDs which appear in test set.

        Returns
        -------
        (train_only, test_only, shared) : Tuple[List]
            train(test)_only is a sorted list of playerIDs which appears only in trian(test) set.
            shared is a sorted list of playerIDs which appears both in train and test set.
        """
        ids_train = set(ids_train.tolist())
        ids_test = set(ids_test.tolist())
        shared = ids_train & ids_test
        train_only = ids_train - shared
        test_only = ids_test - shared
        return sorted(list(train_only)), sorted(list(test_only)), sorted(list(shared))


class Hand(object):
    LEFT = 1
    RIGHT = 0

    @classmethod
    def is_pitcher_hand_left(cls, pitchers: pd.DataFrame) -> pd.Series:
        pitcher_ids = pitchers.pitcherID.unique()
        mode_list = [
            cls.LEFT
            if pitchers.query(f'pitcherID == {id_}').pitcherHand.mode()[0] == 'L' else cls.RIGHT
            for id_ in pitcher_ids]
        most_frequent_hand = dict(zip(pitcher_ids, mode_list))
        return pitchers \
            .pitcherID \
            .apply(lambda id_: most_frequent_hand[id_]) \
            .rename('isPitcherHandLeft')

    @classmethod
    def is_batter_hand_left(cls, batters: pd.DataFrame) -> pd.Series:

        def _impute_batter_hand(
                s: pd.Series,
                num_unique_hand: dict,
                most_frequent_hand: dict,
                left: int,
                right: int) -> int:
            num_unique = num_unique_hand[s.batterID]
            if num_unique == 1:  # one of right or left
                most_frequent_hand = most_frequent_hand[s.batterID][0]
                return right if most_frequent_hand == 'R' else left
            else:  # both
                return right if s.isPitcherHandLeft == left else left

        num_unique_hand = dict(batters.groupby('batterID').batterHand.nunique())
        batter_ids = batters.batterID.unique()
        mode_list = [batters.query(f'batterID == {id_}').batterHand.mode() for id_ in batter_ids]
        most_frequent_hand = dict(zip(batter_ids, mode_list))
        impute_func = partial(
            _impute_batter_hand,
            num_unique_hand=num_unique_hand,
            most_frequent_hand=most_frequent_hand,
            left=cls.LEFT,
            right=cls.RIGHT)
        return batters.apply(impute_func, axis=1).rename('isBatterHandLeft')


class GameParticipation(object):

    def __init__(self, data: pd.DataFrame):
        self.data = data[['gameID', 'startDayTime', 'pitcherID', 'batterID']]

    def hours_elapsed_from_last(self, calc_pitcher: bool = True) -> pd.DataFrame:
        if calc_pitcher:
            data = self.data[['gameID', 'startDayTime', 'pitcherID']] \
                .rename(columns={'pitcherID': 'playerID'})
        else:
            data = self.data[['gameID', 'startDayTime', 'batterID']] \
                .rename(columns={'batterID': 'playerID'})
        data = data.drop_duplicates(subset=['gameID', 'playerID'])
        data['startDayTime'] = pd.to_datetime(data.startDayTime)
        # calculate interval by `playerID`
        list_startdaytime = []
        sec_to_hour = 3600
        for _, player_df in data.groupby('playerID'):
            player_df.sort_values('startDayTime', inplace=True)
            player_df['Previous'] = player_df.startDayTime.shift(1)
            player_df['Delta'] = player_df.startDayTime - player_df.Previous
            player_df['hoursElapsed'] = player_df \
                .Delta \
                .apply(lambda x: x.total_seconds() / sec_to_hour)
            player_df['numGamesParticipated'] = player_df.reset_index().index + 1
            list_startdaytime.append(player_df)
        out_df = pd.concat(list_startdaytime) \
            .sort_values(['playerID', 'gameID']) \
            .loc[:, ['playerID', 'gameID', 'hoursElapsed', 'numGamesParticipated']] \
            .reset_index(drop=True)
        if calc_pitcher:
            out_df.rename(columns={'playerID': 'pitcherID'}, inplace=True)
        else:
            out_df.rename(columns={'playerID': 'batterID'}, inplace=True)
        return out_df
