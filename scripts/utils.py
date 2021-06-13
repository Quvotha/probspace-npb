import os
import random
from typing import Tuple

import numpy as np
import pandas as pd


class DataFrame(object):
    __SOURCE_DF_COLUMN = '__SOURCE__'

    @classmethod
    def merge(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        assert(cls.__SOURCE_DF_COLUMN not in df1.columns.tolist())
        assert(cls.__SOURCE_DF_COLUMN not in df2.columns.tolist())
        df1['cls.__SOURCE_DF_COLUMN'] = 1
        df2['cls.__SOURCE_DF_COLUMN'] = 2
        return pd.concat([df1, df2], axis=0)

    def purge(cls, merged_df: pd.DataFrame) -> pd.DataFrame:
        assert(cls.__SOURCE_DF_COLUMN in merged_df.columns.tolist())
        return (
            merged_df.query(f'{cls.__SOURCE_DF_COLUMN} == 1'),
            merged_df.query(f'{cls.__SOURCE_DF_COLUMN} == 2'),
        )


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
