

import pandas as pd

NO_RECORD = '__NO_DATA__'


def ballXY(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        data=df.ballX.apply(str) + df.ballY,
        name='ballXY'
    )


def extract_patterns(df: pd.DataFrame) -> dict:
    assert(df.gameID.nunique() == 1)
    assert(df.inning.nunique() == 1)
    assert(df.pitcherID.nunique() == 1)
    assert(df.batterID.nunique() == 1)
    assert(df.O.nunique() == 1)

    # Extract values in `totalPitchingCount` order.
    # If there are missing `totalPitchingCount` between 1 and last `totalPitchingCount`,
    # the value there should be filled with a specific value.
    out = {}
    max_count = df.totalPitchingCount.max()
    target_columns = ('ballPositionLabel', 'pitchType', 'ballXY')
    for c in target_columns:
        mapping = dict(
            zip(
                df.totalPitchingCount.tolist(),
                df[c].tolist(),
            )
        )
        orderd_values = [NO_RECORD if i not in mapping.keys() else mapping[i]
                         for i in range(1, max_count + 1)]
        out[c] = ' '.join(orderd_values)
    return out
