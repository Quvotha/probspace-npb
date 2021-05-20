import enum

import pandas as pd


@enum.unique
class Teams(enum.IntEnum):
    HIROSHIMA = 1
    CHUNICHI = 2
    HANSHIN = 3
    KYOJIN = 4
    DeNA = 5
    YAKURUTO = 6
    LOTTE = 7
    RAKUTEN = 8
    NICHIHAMU = 9
    ORIX = 10
    SOFTBANK = 11
    SEIBU = 12

    @classmethod
    def get_team_map(cls) -> dict:
        return {
            'DeNA': Teams.DeNA,
            'ソフトバンク': Teams.SOFTBANK,
            '阪神': Teams.HANSHIN,
            'ヤクルト': Teams.YAKURUTO,
            'オリックス': Teams.ORIX,
            'ロッテ': Teams.LOTTE,
            '楽天': Teams.RAKUTEN,
            '西武': Teams.SEIBU,
            '日本ハム': Teams.NICHIHAMU,
            '中日': Teams.CHUNICHI,
            '広島': Teams.HIROSHIMA,
            '巨人': Teams.KYOJIN,
        }

    @classmethod
    def to_int(cls, team: str) -> int:
        team_map = cls.get_team_map()
        if team not in team_map.keys():
            raise ValueError(f'Cannot convert {team}')
        return team_map[team]

    @classmethod
    def extract_batter_team(cls, s: pd.Series) -> int:
        if s.isBottom == 1:
            return cls.to_int(s.bottomTeam)
        elif s.isBottom == 0:
            return cls.to_int(s.topTeam)
        else:
            raise ValueError(f'`isBottom` must be one of [0, 1] but {s.isBottom} given')

    @classmethod
    def extract_pitcher_team(cls, s: pd.Series) -> int:
        if s.isBottom == 1:
            return cls.to_int(s.topTeam)
        elif s.isBottom == 0:
            return cls.to_int(s.bottomTeam)
        else:
            raise ValueError(f'`isBottom` must be one of [0, 1] but {s.isBottom} given')
