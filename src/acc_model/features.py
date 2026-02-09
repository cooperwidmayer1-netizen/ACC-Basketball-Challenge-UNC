import numpy as np
import pandas as pd
from .config import ROLL_FORM, ROLL_TEMPO, MAX_REST_DAYS

def add_team_features(df):
    df = df.sort_values("date").copy()
    teams = pd.unique(df[["home", "away"]].values.ravel())

    for t in teams:
        mask_h = df["home"] == t
        mask_a = df["away"] == t

        margins = np.where(
            mask_h, df["home_score"] - df["away_score"],
            np.where(mask_a, df["away_score"] - df["home_score"], np.nan)
        )

        tempo = df["home_score"] + df["away_score"]

        df.loc[mask_h | mask_a, f"{t}_form"] = (
            pd.Series(margins).rolling(ROLL_FORM, min_periods=1).mean().values
        )

        df.loc[mask_h | mask_a, f"{t}_tempo"] = (
            tempo.rolling(ROLL_TEMPO, min_periods=1).mean().values
        )

    return df.fillna(0)
