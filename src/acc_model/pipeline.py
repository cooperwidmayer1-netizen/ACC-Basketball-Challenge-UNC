from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from .config import (
    ALPHA_GRID,
    BASIC_ALPHA,
    BLEND_W_GRID,
    CLIP_GRID,
    FETCH_END_ISO,
    FETCH_START_ISO,
    FORM_SCALE_GRID,
    FORM_W_GRID,
    MANUAL_GAMES,
    PREDICT_FILTER,
    PRED_END,
    PRED_START,
    SLEEP_S,
    SLOPE_SHRINK_GRID,
    TRAIN_CUTOFF,
    VAL_END,
    VAL_START,
    WINSOR_GRID,
)
from .espn import fetch_and_prepare_games, make_game_id
from .features import (
    add_no_leak_form_tempo,
    attach_frozen_form_tempo_future,
    freeze_team_form_tempo_at_cutoff,
)
from .models import (
    X_points_adv,
    X_spread_basic,
    build_adv_team_index,
    build_basic_team_index,
    fit_adv_and_get_preds,
)
from .names import ACC_TEAMS, norm_team
from .schedule import build_rest_from_schedule, load_csv_authority_schedule, merge_csv_with_espn_future


def run(csv_schedule: str, out_pred: str, out_tuning: str) -> None:
    """
    Run the notebook-faithful full pipeline:
    - fetch ESPN season
    - build train/validation/future sets
    - tune advanced model pre-cutoff
    - refit once on all pre-cutoff data
    - predict future games
    """
    out_pred_path = Path(out_pred)
    out_tuning_path = Path(out_tuning)
    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_tuning_path.parent.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1) Fetch + prep
    # ============================================================
    df = fetch_and_prepare_games(FETCH_START_ISO, FETCH_END_ISO, sleep_s=SLEEP_S)

    assert "game_id" in df.columns, "game_id missing (dedup not applied?)"
    assert df["game_id"].notna().all(), "Some game_id are NaN"

    df["home_ACC"] = df["home_team_short"].isin(ACC_TEAMS)
    df["away_ACC"] = df["away_team_short"].isin(ACC_TEAMS)
    df["acc_vs_acc"] = df["home_ACC"] & df["away_ACC"]
    df["acc_involved"] = df["home_ACC"] | df["away_ACC"]

    df_scored = df.dropna(subset=["home_score", "away_score"]).copy()
    past = df_scored[df_scored["date"] < TRAIN_CUTOFF].copy().reset_index(drop=True)

    future_window = df[(df["date"] >= PRED_START) & (df["date"] < PRED_END)].copy().reset_index(drop=True)
    future_espn_acc = future_window[future_window["acc_vs_acc"]].copy()

    future_csv = load_csv_authority_schedule(csv_schedule, PRED_START, PRED_END)
    future_pred = merge_csv_with_espn_future(future_csv, future_espn_acc)

    # ============================================================
    # MANUAL NON-ACC GAMES
    # ============================================================
    manual_games = pd.DataFrame(MANUAL_GAMES)
    manual_games["date"] = pd.to_datetime(manual_games["date"], utc=True)
    manual_games["neutral_site"] = False
    manual_games["home_team_short"] = manual_games["home_team_short"].apply(norm_team)
    manual_games["away_team_short"] = manual_games["away_team_short"].apply(norm_team)
    manual_games["game_id"] = make_game_id(manual_games)

    future_pred = (
        pd.concat([future_pred, manual_games], ignore_index=True)
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    print("Future games AFTER manual add:", len(future_pred))
    print(f"\nTRAIN scored games (< {TRAIN_CUTOFF.date()}): {len(past)}")
    print(
        f"FUTURE games to predict "
        f"({PRED_START.date()}–{(PRED_END - pd.Timedelta(days=1)).date()}, "
        f"filter={PREDICT_FILTER}): {len(future_pred)}"
    )

    # Build rest map on augmented schedule
    sched_cols = ["game_id", "date", "home_team_short", "away_team_short", "neutral_site"]
    df_sched = df[sched_cols].copy()

    missing_ids = set(future_pred["game_id"]) - set(df_sched["game_id"])
    csv_only_sched = future_pred[future_pred["game_id"].isin(missing_ids)][sched_cols].copy()

    aug_sched = pd.concat([df_sched, csv_only_sched], ignore_index=True)
    aug_sched = (
        aug_sched.sort_values(["date", "game_id"])
        .drop_duplicates("game_id", keep="first")
        .reset_index(drop=True)
    )

    rest_map_all = build_rest_from_schedule(aug_sched)

    # ============================================================
    # 2) Train BASIC (pre-cutoff only)
    # ============================================================
    train_basic = past[past["acc_involved"]].copy().reset_index(drop=True)
    teams_basic, idx_basic = build_basic_team_index(train_basic)
    n_basic = len(teams_basic)

    Xtr_b = X_spread_basic(train_basic, idx_basic=idx_basic, n_basic=n_basic)
    ytr_b = (train_basic["home_score"] - train_basic["away_score"]).to_numpy(float)
    basic = Ridge(alpha=BASIC_ALPHA, fit_intercept=True).fit(Xtr_b, ytr_b)

    # ============================================================
    # 3) ADV model setup + tuning (pre-cutoff only, NO-LEAK)
    # ============================================================
    train_full = past[past["acc_involved"]].copy().reset_index(drop=True)

    val_raw = train_full[
        (train_full["date"] >= VAL_START) & (train_full["date"] < VAL_END)
    ].copy().reset_index(drop=True)
    trn_raw = train_full[train_full["date"] < VAL_START].copy().reset_index(drop=True)

    print(f"\nTuning split (pre-cutoff only): train={len(trn_raw)} | val={len(val_raw)}")

    teams_adv, idx_adv = build_adv_team_index(train_full)
    n_adv = len(teams_adv)

    val_teams = set(val_raw["home_team_short"]) | set(val_raw["away_team_short"])
    missing_adv = sorted([t for t in val_teams if t not in idx_adv])
    if missing_adv:
        print("⚠️ ADV unseen teams in val (will be treated as zeros):", missing_adv)

    X_val_basic = X_spread_basic(val_raw, idx_basic=idx_basic, n_basic=n_basic)
    pred_val_basic = basic.predict(X_val_basic)
    y_val_true = (val_raw["home_score"] - val_raw["away_score"]).to_numpy(float)
    val_mae_basic = mean_absolute_error(y_val_true, pred_val_basic)
    print(f"\nVAL MAE (BASIC only): {val_mae_basic:.4f}")

    best = None
    rows: list[dict] = []

    scored_pool_pre_cutoff = df_scored[df_scored["date"] < TRAIN_CUTOFF].copy()

    for form_w in FORM_W_GRID:
        trn_base = trn_raw.merge(rest_map_all, on="game_id", how="left")
        val_base = val_raw.merge(rest_map_all, on="game_id", how="left")

        trn_feat0 = add_no_leak_form_tempo(trn_base, scored_pool_pre_cutoff, form_w=form_w, min_p=2)
        val_feat0 = add_no_leak_form_tempo(val_base, scored_pool_pre_cutoff, form_w=form_w, min_p=2)

        for form_scale in FORM_SCALE_GRID:
            trn = trn_feat0.copy()
            val = val_feat0.copy()

            trn["home_form_margin"] *= float(form_scale)
            trn["away_form_margin"] *= float(form_scale)
            trn["form_diff"] = trn["home_form_margin"] - trn["away_form_margin"]

            val["home_form_margin"] *= float(form_scale)
            val["away_form_margin"] *= float(form_scale)
            val["form_diff"] = val["home_form_margin"] - val["away_form_margin"]

            for (wlo, whi) in WINSOR_GRID:
                for alpha in ALPHA_GRID:
                    cal_a, cal_b, pred_val_adv_raw = fit_adv_and_get_preds(
                        trn,
                        val,
                        wlo=wlo,
                        whi=whi,
                        alpha_points=alpha,
                        idx_adv=idx_adv,
                        n_adv=n_adv,
                    )

                    for s in SLOPE_SHRINK_GRID:
                        b_adj = (1.0 - float(s)) * float(cal_b) + float(s) * 1.0
                        pred_val_adv_cal = float(cal_a) + b_adj * pred_val_adv_raw

                        for w in BLEND_W_GRID:
                            pred_val_mix = float(w) * pred_val_adv_cal + (1.0 - float(w)) * pred_val_basic

                            for C in CLIP_GRID:
                                pred_val_final = np.clip(pred_val_mix, -float(C), float(C))
                                mae = mean_absolute_error(y_val_true, pred_val_final)

                                row = {
                                    "form_w": int(form_w),
                                    "form_scale": float(form_scale),
                                    "winsor": f"{wlo:.3f}-{whi:.3f}",
                                    "alpha": float(alpha),
                                    "slope_shrink": float(s),
                                    "blend_w": float(w),
                                    "clip_C": float(C),
                                    "val_mae": float(mae),
                                    "cal_a": float(cal_a),
                                    "cal_b": float(cal_b),
                                    "b_adj": float(b_adj),
                                }
                                rows.append(row)

                                if best is None or mae < best["val_mae"]:
                                    best = row

    tune = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    tune.to_csv(out_tuning_path, index=False)

    print("\nTop 20 configs by VAL MAE (pre-cutoff only):")
    print(tune.head(20).to_string(index=False))
    print("\n✅ Selected ONE-SHOT config:", best)
    print(f"BEST VAL MAE (final post-processed): {best['val_mae']:.4f}")

    # Recompute ADV-only calibrated val MAE for selected config
    best_form_w = int(best["form_w"])
    best_form_scale = float(best["form_scale"])
    best_wlo, best_whi = map(float, best["winsor"].split("-"))
    best_alpha = float(best["alpha"])
    best_s = float(best["slope_shrink"])

    trn_base = trn_raw.merge(rest_map_all, on="game_id", how="left")
    val_base = val_raw.merge(rest_map_all, on="game_id", how="left")
    trn_feat0 = add_no_leak_form_tempo(trn_base, scored_pool_pre_cutoff, form_w=best_form_w, min_p=2)
    val_feat0 = add_no_leak_form_tempo(val_base, scored_pool_pre_cutoff, form_w=best_form_w, min_p=2)

    trn_feat0["home_form_margin"] *= best_form_scale
    trn_feat0["away_form_margin"] *= best_form_scale
    trn_feat0["form_diff"] = trn_feat0["home_form_margin"] - trn_feat0["away_form_margin"]

    val_feat0["home_form_margin"] *= best_form_scale
    val_feat0["away_form_margin"] *= best_form_scale
    val_feat0["form_diff"] = val_feat0["home_form_margin"] - val_feat0["away_form_margin"]

    cal_a, cal_b, pred_val_adv_raw = fit_adv_and_get_preds(
        trn_feat0,
        val_feat0,
        wlo=best_wlo,
        whi=best_whi,
        alpha_points=best_alpha,
        idx_adv=idx_adv,
        n_adv=n_adv,
    )
    b_adj = (1.0 - best_s) * float(cal_b) + float(best_s) * 1.0
    pred_val_adv_cal = float(cal_a) + float(b_adj) * pred_val_adv_raw
    val_mae_adv_cal = mean_absolute_error(y_val_true, pred_val_adv_cal)
    print(f"VAL MAE (ADV calibrated only; no blend/clip): {val_mae_adv_cal:.4f}")

    # ============================================================
    # 4) Refit ADV once on ALL pre-cutoff ACC-involved with best config
    # ============================================================
    best_blend_w = float(best["blend_w"])
    best_clip_C = float(best["clip_C"])

    train_adv = train_full.merge(rest_map_all, on="game_id", how="left")
    train_adv = add_no_leak_form_tempo(train_adv, scored_pool_pre_cutoff, form_w=best_form_w, min_p=2)
    train_adv["home_form_margin"] *= best_form_scale
    train_adv["away_form_margin"] *= best_form_scale
    train_adv["form_diff"] = train_adv["home_form_margin"] - train_adv["away_form_margin"]

    train_adv_w = train_adv.copy()
    for col in ["home_score", "away_score"]:
        lo, hi = train_adv_w[col].quantile([best_wlo, best_whi])
        train_adv_w[col] = train_adv_w[col].clip(lo, hi)

    Xh_tr = X_points_adv(train_adv_w, home=True, idx_adv=idx_adv, n_adv=n_adv)
    Xa_tr = X_points_adv(train_adv_w, home=False, idx_adv=idx_adv, n_adv=n_adv)
    yh_tr = train_adv_w["home_score"].to_numpy(float)
    ya_tr = train_adv_w["away_score"].to_numpy(float)

    m_home = Ridge(alpha=float(best_alpha), fit_intercept=False).fit(Xh_tr, yh_tr)
    m_away = Ridge(alpha=float(best_alpha), fit_intercept=False).fit(Xa_tr, ya_tr)

    pred_tr_raw = m_home.predict(Xh_tr) - m_away.predict(Xa_tr)
    y_spread_tr = (train_adv_w["home_score"] - train_adv_w["away_score"]).to_numpy(float)

    Z = np.vstack([np.ones_like(pred_tr_raw), pred_tr_raw]).T
    cal = Ridge(alpha=0.0, fit_intercept=False).fit(Z, y_spread_tr)
    cal_a, cal_b = cal.coef_
    b_adj = (1.0 - best_s) * float(cal_b) + float(best_s) * 1.0

    print(f"\nFinal calibration (pre-cutoff): a={float(cal_a):.4f}, b={float(cal_b):.4f}, b_adj={b_adj:.4f}")
    print(
        f"Post-process: blend_w={best_blend_w:.2f}, "
        f"clip_C={best_clip_C:.0f}, form_scale={best_form_scale:.2f}"
    )

    # ============================================================
    # 5) Predict future games with frozen features
    # ============================================================
    X_future_basic = X_spread_basic(future_pred, idx_basic=idx_basic, n_basic=n_basic)
    pred_basic_future = basic.predict(X_future_basic)

    team_form, team_tempo, form_fb, tempo_fb = freeze_team_form_tempo_at_cutoff(
        scored_games=scored_pool_pre_cutoff,
        cutoff=TRAIN_CUTOFF,
        form_w=best_form_w,
        min_p=2,
    )

    future_feat = future_pred.merge(rest_map_all, on="game_id", how="left")
    for c in ["home_rest_days", "away_rest_days", "rest_diff"]:
        if c not in future_feat.columns:
            future_feat[c] = 0.0
        future_feat[c] = future_feat[c].fillna(0.0)

    future_feat = attach_frozen_form_tempo_future(
        future_feat,
        team_form,
        team_tempo,
        form_fb,
        tempo_fb,
        form_scale=best_form_scale,
    )

    Xh_fu = X_points_adv(future_feat, home=True, idx_adv=idx_adv, n_adv=n_adv)
    Xa_fu = X_points_adv(future_feat, home=False, idx_adv=idx_adv, n_adv=n_adv)
    pred_adv_raw_future = m_home.predict(Xh_fu) - m_away.predict(Xa_fu)
    pred_adv_cal_future = float(cal_a) + float(b_adj) * pred_adv_raw_future

    pred_final = best_blend_w * pred_adv_cal_future + (1.0 - best_blend_w) * pred_basic_future
    pred_final = np.clip(pred_final, -best_clip_C, best_clip_C)

    submit = future_feat[["date", "home_team_short", "away_team_short", "neutral_site"]].copy()
    submit["pred_basic_spread"] = pred_basic_future
    submit["pred_adv_spread"] = pred_adv_cal_future
    submit["pred_final_spread"] = pred_final
    submit = submit.sort_values("date").reset_index(drop=True)

    print("\nSUBMISSION PREVIEW:")
    print(submit.head(min(78, len(submit))).to_string(index=False))

    print("\nCounts check:")
    print("Train games:", len(past))
    print("Validation games:", len(val_raw))
    print("Future games:", len(future_pred))
    print("Any NaN preds?", submit["pred_final_spread"].isna().any())
    print("Date range:", submit["date"].min(), "→", submit["date"].max())

    submit.to_csv(out_pred_path, index=False)
    print(f"\nSaved predictions: {out_pred_path}")
    print(f"Saved tuning table: {out_tuning_path}")
