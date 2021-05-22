from functools import partial

import numpy as np
import pandas as pd


class Players(object):

    # These players belonged to 2 teams in 2020s.
    # https://prob.space/competitions/npb/discussions/Quvotha-nndropout100-Post996ddd42670380ebceb5
    TRANSFERRED_PLAYERS = ('ＤＪ．ジョンソン', '澤村 拓一', '小林 慶祐')

    def __init__(self, pitchers: pd.DataFrame, batters: pd.DataFrame):
        self.pitchers = pitchers[['pitcher', 'pitcherTeam']]
        self.batters = batters[['batter', 'batterTeam']]

    def assign_id(self) -> pd.DataFrame:
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
            list_startdaytime.append(player_df)
        out_df = pd.concat(list_startdaytime) \
            .sort_values(['playerID', 'gameID']) \
            .loc[:, ['playerID', 'gameID', 'hoursElapsed']] \
            .reset_index(drop=True)
        if calc_pitcher:
            out_df.rename(columns={'playerID': 'pitcherID'}, inplace=True)
        else:
            out_df.rename(columns={'playerID': 'batterID'}, inplace=True)
        return out_df
