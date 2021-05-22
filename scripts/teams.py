import pandas as pd


class Teams(object):

    @staticmethod
    def extract_batter_team(s: pd.Series) -> int:
        if s.isBottom == 1:
            return s.bgBottom
        elif s.isBottom == 0:
            return s.bgTop
        else:
            raise ValueError(f'`isBottom` must be one of [0, 1] but {s.isBottom} given')

    @staticmethod
    def extract_pitcher_team(s: pd.Series) -> int:
        if s.isBottom == 1:
            return s.bgTop
        elif s.isBottom == 0:
            return s.bgBottom
        else:
            raise ValueError(f'`isBottom` must be one of [0, 1] but {s.isBottom} given')
