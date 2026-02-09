import numpy as np
import pandas as pd
from .espn import fetch_espn
from .schedule import load_authority_schedule, merge_schedule
from .features import add_team_features
from .models import fit_ridge, capped_mae
from .config import *

def run(csv_schedule, out_pred, out_tuning):
    fetched = fetch_espn(FETCH_START_ISO, FETCH_END_ISO)
    sched = load_authority_schedule(csv_schedule)
    df = merge_schedule(fetched, sched)

    df["margin"] = df["home_score"] - df["away_score"]
    df = add_team_features(df)

    train = df[(df["date"] < TRAIN_CUTOFF) & df["margin"].notna()]
    test  = df[df["date"] >= TRAIN_CUTOFF]

    feature_cols = [c for c in df.columns if c.endswith("_form") or c.endswith("_tempo")]

    X_train = train[feature_cols].values
    y_train = train["margin"].values
    X_test  = test[feature_cols].values

    rows = []
    best = None

    for a in ALPHAS:
        m = fit_ridge(X_train, y_train, a)
        pred = m.predict(X_train)
        score = capped_mae(y_train, pred)
        rows.append({"alpha": a, "mae": score})
        if best is None or score < best[0]:
            best = (score, m)

    pd.DataFrame(rows).to_csv(out_tuning, index=False)

    test["pred_spread"] = best[1].predict(X_test)
    test[["date", "home", "away", "pred_spread"]].to_csv(out_pred, index=False)
