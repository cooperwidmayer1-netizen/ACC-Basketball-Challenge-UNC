from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def build_basic_team_index(train_basic: pd.DataFrame):
    teams = sorted(set(train_basic["home_team_short"]) | set(train_basic["away_team_short"]))
    idx = {t: i for i, t in enumerate(teams)}
    return teams, idx


def X_spread_basic(d: pd.DataFrame, idx_basic: dict[str, int], n_basic: int) -> np.ndarray:
    """
    Basic spread model design matrix from the notebook.
    """
    X = np.zeros((len(d), 2 * n_basic + 1), dtype=float)

    for i, r in enumerate(d.itertuples(index=False)):
        h = idx_basic.get(r.home_team_short, None)
        a = idx_basic.get(r.away_team_short, None)

        if h is not None:
            X[i, h] = 1
            X[i, n_basic + h] = -1
        if a is not None:
            X[i, a] = -1
            X[i, n_basic + a] = 1

        X[i, -1] = 0.0 if bool(r.neutral_site) else 1.0

    return X


def build_adv_team_index(train_full: pd.DataFrame):
    teams = sorted(set(train_full["home_team_short"]) | set(train_full["away_team_short"]))
    idx = {t: i for i, t in enumerate(teams)}
    return teams, idx


def X_points_adv(
    df_in: pd.DataFrame,
    home: bool,
    idx_adv: dict[str, int],
    n_adv: int,
) -> np.ndarray:
    """
    Advanced points model design matrix from the notebook.
    """
    p = 3 * n_adv + 1 + 3 + 3 + 1
    X = np.zeros((len(df_in), p), dtype=float)

    for i, r in enumerate(df_in.itertuples(index=False)):
        h = idx_adv.get(r.home_team_short, None)
        a = idx_adv.get(r.away_team_short, None)

        if home:
            if h is not None:
                X[i, h] = 1.0
            if a is not None:
                X[i, n_adv + a] = 1.0
            if h is not None:
                X[i, 2 * n_adv + h] = 0.0 if bool(r.neutral_site) else 1.0
        else:
            if a is not None:
                X[i, a] = 1.0
            if h is not None:
                X[i, n_adv + h] = 1.0

        X[i, 3 * n_adv + 0] = float(r.tempo)

        base = 3 * n_adv + 1
        X[i, base + 0] = float(r.home_rest_days)
        X[i, base + 1] = float(r.away_rest_days)
        X[i, base + 2] = float(r.rest_diff)

        base2 = base + 3
        X[i, base2 + 0] = float(r.home_form_margin)
        X[i, base2 + 1] = float(r.away_form_margin)
        X[i, base2 + 2] = float(r.form_diff)

        X[i, -1] = 1.0

    return X


def fit_adv_and_get_preds(
    train_scored: pd.DataFrame,
    val_scored: pd.DataFrame,
    wlo: float,
    whi: float,
    alpha_points: float,
    idx_adv: dict[str, int],
    n_adv: int,
):
    """
    Fit notebook advanced home/away points models and return calibration + raw val predictions.
    """
    tr = train_scored.copy()
    va = val_scored.copy()

    tr_w = tr.copy()
    for col in ["home_score", "away_score"]:
        lo, hi = tr_w[col].quantile([wlo, whi])
        tr_w[col] = tr_w[col].clip(lo, hi)

    Xh_tr = X_points_adv(tr_w, home=True, idx_adv=idx_adv, n_adv=n_adv)
    Xa_tr = X_points_adv(tr_w, home=False, idx_adv=idx_adv, n_adv=n_adv)
    yh_tr = tr_w["home_score"].to_numpy(float)
    ya_tr = tr_w["away_score"].to_numpy(float)

    m_home = Ridge(alpha=float(alpha_points), fit_intercept=False).fit(Xh_tr, yh_tr)
    m_away = Ridge(alpha=float(alpha_points), fit_intercept=False).fit(Xa_tr, ya_tr)

    pred_tr = m_home.predict(Xh_tr) - m_away.predict(Xa_tr)
    y_spread_tr = (tr_w["home_score"] - tr_w["away_score"]).to_numpy(float)

    Z = np.vstack([np.ones_like(pred_tr), pred_tr]).T
    cal = Ridge(alpha=0.0, fit_intercept=False).fit(Z, y_spread_tr)
    cal_a, cal_b = cal.coef_

    Xh_va = X_points_adv(va, home=True, idx_adv=idx_adv, n_adv=n_adv)
    Xa_va = X_points_adv(va, home=False, idx_adv=idx_adv, n_adv=n_adv)
    pred_va_raw = m_home.predict(Xh_va) - m_away.predict(Xa_va)

    return float(cal_a), float(cal_b), pred_va_raw
