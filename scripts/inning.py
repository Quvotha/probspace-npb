import pandas as pd


class Inning(object):

    @staticmethod
    def extract_info(inning: str) -> pd.Series:
        inning_info = inning.split('回')
        if len(inning_info) != 2:
            raise ValueError(f"`inning` format is invalid: {inning}")
        else:
            no = int(inning_info[0])
            if inning_info[1] == "裏":
                is_bottom = 1
            elif inning_info[1] == "表":
                is_bottom = 0
            else:
                raise ValueError(f"`inning` format is invalid: {inning}")
        return pd.Series({'inningNo': no, 'isBottom': is_bottom})
