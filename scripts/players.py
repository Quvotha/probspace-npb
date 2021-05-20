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
    def impute_pitcher_hand(cls, pitchers: pd.DataFrame) -> dict:
        impute_values = {}
        nan_mask = pitchers.pitcherHand.apply(lambda x: isinstance(x, float))
        for id_ in pitchers[nan_mask].pitcherID.unique():
            hand = pitchers.query(f'pitcherID == {id_}').dropna().pitcherHand.mode()[0]
            impute_values[id_] = cls.LEFT if hand == 'L' else cls.RIGHT
        return impute_values

    @classmethod
    def impute_batter_hand(cls, batters: pd.DataFrame) -> pd.Series:

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
        return batters.apply(impute_func, axis=1)
