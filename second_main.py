# main.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data_loader import load_and_clean
from src.elo import EloParams, MarginElo
from src.metrics import mae

# =========================
# SET YOUR YEARS/FILES HERE
# =========================
SEASON_1 = "2022-23"
SEASON_2 = "2023-24"
SEASON_3 = "2024-25"

RAW_1 = "data/raw/acc_2022_23.csv"
RAW_2 = "data/raw/acc_2023_24.csv"
RAW_3 = "data/raw/acc_2024_25.csv"

SCORE_MONTHS = [2, 3]  # evaluate Feb–Mar only

# =========================
# GRID SEARCH RANGES
# =========================
# Base Elo params (we will override K dynamically with two-speed K)
HCA_grid = [55, 60, 70]
SCALE_grid = [10, 12]
CLIP_grid = [4, 6, 8]
GAMMA_grid = [0.60, 0.65, 0.70]          # season shrink strength
N_grid = [2, 3, 5]                 # recent form window (venue-split, opponent-weighted)
PACE_N_grid = [3, 5, 10]           # pace window (venue-split)
PCAP_grid = [7, 9]           # prediction cap
RIDGE_ALPHA_grid = [0.1, 1.0, 5.0] # calibration strength

# Two-speed K grids (NEW)
K_EARLY_grid = [10, 15]
K_LATE_grid = [20, 25]

# ------------------------
# helpers
# ------------------------
def _mean(lst):
    return float(np.mean(lst)) if lst else 0.0

def _push(buf_dict, key, value, maxlen):
    buf_dict.setdefault(key, []).append(float(value))
    if len(buf_dict[key]) > maxlen:
        buf_dict[key] = buf_dict[key][-maxlen:]


def season_to_year(season_label: str) -> int:
    """
    "2023-24" -> 2024
    "2023-2024" -> 2024
    """
    try:
        right = season_label.split("-")[1]
        right = right.strip()
        if len(right) == 2:
            return 2000 + int(right)
        return int(right)
    except Exception:
        # fallback: use current year if parsing fails
        return pd.Timestamp.today().year


def k_schedule(game_date: pd.Timestamp, season_label: str, k_early: float, k_late: float) -> float:
    """
    Use K_early before Jan 15 of the spring year; K_late on/after Jan 15.
    """
    y = season_to_year(season_label)
    cutoff = pd.Timestamp(year=y, month=1, day=15)
    return k_early if game_date < cutoff else k_late


def opp_weight(opp_elo: float) -> float:
    """
    Opponent-strength weight. Beating strong teams counts a bit more; weak teams a bit less.
    Clip keeps it stable.
    """
    w = 1.0 + (opp_elo - 1500.0) / 400.0
    return float(np.clip(w, 0.75, 1.25))


# ------------------------
# simulate pre-game features (NO leakage; update AFTER result)
# Adds:
#   - two-speed K updates (K_early vs K_late)
#   - opponent-weighted recent margins (venue-split)
# Keeps:
#   - venue-split recent form
#   - venue-split pace proxy
#   - season shrink + reset buffers at season boundaries
# ------------------------
def simulate_with_features(
    df: pd.DataFrame,
    model: MarginElo,
    recent_window: int,
    pace_window: int,
    update_ratings: bool,
    k_early: float,
    k_late: float,
    do_season_shrink: bool = False,
    season_col: str | None = None,
):
    df = df.sort_values("date").reset_index(drop=True).copy()

    # recent margins split by venue (opponent-weighted)
    home_margin_buf = {}  # team -> list of margins in HOME games (weighted)
    away_margin_buf = {}  # team -> list of margins in AWAY games (weighted)

    # pace proxy (total points) split by venue
    home_pace_buf = {}    # team -> list of totals in HOME games
    away_pace_buf = {}    # team -> list of totals in AWAY games

    elo_preds = []
    recent_venue_diffs = []
    pace_diffs = []

    last_season = None

    for r in df.itertuples(index=False):
        # season boundary handling
        if do_season_shrink and season_col is not None:
            cur_season = getattr(r, season_col)
            if last_season is None:
                last_season = cur_season
            elif cur_season != last_season:
                model.shrink_to_mean()  # uses params.shrink_gamma
                # reset buffers (new season, new roster context)
                home_margin_buf, away_margin_buf = {}, {}
                home_pace_buf, away_pace_buf = {}, {}
                last_season = cur_season

        home, away = r.home_team, r.away_team
        d = r.date
        season_label = getattr(r, season_col) if season_col is not None else ""

        # --- pre-game Elo-based spread ---
        elo_hat = model.predict(home, away)

        # --- pre-game venue-split recent form (already opponent-weighted in buffers) ---
        home_recent_home = _mean(home_margin_buf.get(home, []))
        away_recent_away = _mean(away_margin_buf.get(away, []))
        recent_venue_diff = home_recent_home - away_recent_away

        # --- pre-game venue-split pace proxy ---
        home_pace = _mean(home_pace_buf.get(home, []))
        away_pace = _mean(away_pace_buf.get(away, []))
        pace_diff = home_pace - away_pace

        elo_preds.append(float(elo_hat))
        recent_venue_diffs.append(float(recent_venue_diff))
        pace_diffs.append(float(pace_diff))

        if update_ratings:
            actual_margin = float(r.spread)
            total_pts = float(r.home_pts + r.away_pts)

            # opponent Elo BEFORE update (for weighting)
            home_elo_pre = model.elo.get(home, 1500.0)
            away_elo_pre = model.elo.get(away, 1500.0)

            # opponent-weighted margins (team perspective)
            w_home = opp_weight(away_elo_pre)   # home margin weighted by away strength
            w_away = opp_weight(home_elo_pre)   # away margin weighted by home strength

            # update buffers AFTER game (no leakage)
            _push(home_margin_buf, home, actual_margin * w_home, recent_window)
            _push(away_margin_buf, away, (-actual_margin) * w_away, recent_window)

            _push(home_pace_buf, home, total_pts, pace_window)
            _push(away_pace_buf, away, total_pts, pace_window)

            # choose K (two-speed schedule)
            k_use = k_schedule(d, season_label, k_early=k_early, k_late=k_late)

            # update Elo AFTER game with chosen K
            # NOTE: requires src/elo.py to have MarginElo.update_with_k(...)
            model.update_with_k(home, away, actual_margin, k_override=k_use)

    df["elo_pred"] = np.array(elo_preds, dtype=float)
    df["recent_venue_diff"] = np.array(recent_venue_diffs, dtype=float)
    df["pace_diff"] = np.array(pace_diffs, dtype=float)
    return df


# ------------------------
# tuning evaluation: train on S1, score S2 Feb–Mar
# ------------------------
def eval_params_on_s1_to_s2(
    df_s1_full, df_s2_full, params,
    n_recent, n_pace, pred_cap, ridge_alpha,
    k_early, k_late
):
    all_teams = pd.unique(pd.concat([df_s1_full, df_s2_full])[["home_team", "away_team"]].values.ravel())
    model = MarginElo(teams=all_teams, params=params)

    s1 = df_s1_full.copy(); s1["season"] = SEASON_1
    s2 = df_s2_full.copy(); s2["season"] = SEASON_2
    full = pd.concat([s1, s2], ignore_index=True).sort_values("date").reset_index(drop=True)

    walk = simulate_with_features(
        df=full,
        model=model,
        recent_window=n_recent,
        pace_window=n_pace,
        update_ratings=True,
        k_early=k_early,
        k_late=k_late,
        do_season_shrink=True,
        season_col="season",
    )

    # calibrate on S1 only (no peeking into S2)
    train_part = walk[walk["season"] == SEASON_1]
    X_tr = train_part[["elo_pred", "recent_venue_diff", "pace_diff"]].values
    y_tr = train_part["spread"].values
    cal = Ridge(alpha=ridge_alpha).fit(X_tr, y_tr)

    # score S2 Feb–Mar
    val_part = walk[(walk["season"] == SEASON_2) & (walk["date"].dt.month.isin(SCORE_MONTHS))].copy()
    X_val = val_part[["elo_pred", "recent_venue_diff", "pace_diff"]].values
    pred = cal.predict(X_val)
    pred = np.clip(pred, -pred_cap, pred_cap)

    return mae(val_part["spread"].values, pred)


def grid_search(df_s1_full, df_s2_full):
    best = None
    top = []

    for gamma in GAMMA_grid:
        for n_recent in N_grid:
            for n_pace in PACE_N_grid:
                for pred_cap in PCAP_grid:
                    for ridge_alpha in RIDGE_ALPHA_grid:
                        for k_early in K_EARLY_grid:
                            for k_late in K_LATE_grid:
                                for hca in HCA_grid:
                                    for scale in SCALE_grid:
                                        for clip in CLIP_grid:
                                            # base params: k is unused for updates (we override via update_with_k)
                                            params = EloParams(
                                                k=20,  # placeholder
                                                hca=hca,
                                                scale=scale,
                                                clip=clip,
                                                shrink_gamma=gamma,
                                            )
                                            score = eval_params_on_s1_to_s2(
                                                df_s1_full, df_s2_full,
                                                params=params,
                                                n_recent=n_recent,
                                                n_pace=n_pace,
                                                pred_cap=pred_cap,
                                                ridge_alpha=ridge_alpha,
                                                k_early=k_early,
                                                k_late=k_late,
                                            )
                                            cand = (score, params, n_recent, n_pace, pred_cap, ridge_alpha, k_early, k_late)
                                            top.append(cand)
                                            if best is None or score < best[0]:
                                                best = cand

    top.sort(key=lambda x: x[0])
    return best, top[:10]


# ------------------------
# final test: train on S1+S2, walk S3, score Feb–Mar
# ------------------------
def final_test_on_s3(
    df_s1_full, df_s2_full, df_s3_full,
    best_params, n_recent, n_pace, pred_cap, ridge_alpha,
    k_early, k_late
):
    all_teams = pd.unique(pd.concat([df_s1_full, df_s2_full, df_s3_full])[["home_team", "away_team"]].values.ravel())
    model = MarginElo(teams=all_teams, params=best_params)

    s1 = df_s1_full.copy(); s1["season"] = SEASON_1
    s2 = df_s2_full.copy(); s2["season"] = SEASON_2
    s3 = df_s3_full.copy(); s3["season"] = SEASON_3
    full = pd.concat([s1, s2, s3], ignore_index=True).sort_values("date").reset_index(drop=True)

    walk = simulate_with_features(
        df=full,
        model=model,
        recent_window=n_recent,
        pace_window=n_pace,
        update_ratings=True,
        k_early=k_early,
        k_late=k_late,
        do_season_shrink=True,
        season_col="season",
    )

    # calibrate on S1+S2 only
    train_part = walk[walk["season"].isin([SEASON_1, SEASON_2])]
    X_tr = train_part[["elo_pred", "recent_venue_diff", "pace_diff"]].values
    y_tr = train_part["spread"].values
    cal = Ridge(alpha=ridge_alpha).fit(X_tr, y_tr)

    # test on S3 Feb–Mar
    test_part = walk[(walk["season"] == SEASON_3) & (walk["date"].dt.month.isin(SCORE_MONTHS))].copy()
    X_te = test_part[["elo_pred", "recent_venue_diff", "pace_diff"]].values

    pred_raw = test_part["elo_pred"].values
    pred_cal = cal.predict(X_te)
    pred_cal = np.clip(pred_cal, -pred_cap, pred_cap)

    raw_mae = mae(test_part["spread"].values, pred_raw)
    cal_mae = mae(test_part["spread"].values, pred_cal)

    out = test_part.copy()
    out["pred_raw_elo"] = pred_raw
    out["pred_calibrated"] = pred_cal
    return raw_mae, cal_mae, out


def main():
    df_s1 = load_and_clean(RAW_1, months=None)
    df_s2 = load_and_clean(RAW_2, months=None)
    df_s3 = load_and_clean(RAW_3, months=None)

    best, top10 = grid_search(df_s1, df_s2)

    print(f"\nTop 10 (tuning on {SEASON_2} Feb–Mar calibrated MAE):")
    for score, p, n_recent, n_pace, pcap, alpha, k_early, k_late in top10:
        print(
            f"MAE={score:.3f} | HCA={p.hca} SCALE={p.scale} CLIP={p.clip} GAMMA={p.shrink_gamma} "
            f"| N_RECENT={n_recent} N_PACE={n_pace} PCAP={pcap} RIDGE={alpha} "
            f"| K_EARLY={k_early} K_LATE={k_late}"
        )

    best_score, best_params, n_recent, n_pace, pcap, alpha, k_early, k_late = best
    print("\nBest tuned:")
    print(
        f"MAE={best_score:.3f} | HCA={best_params.hca} SCALE={best_params.scale} "
        f"CLIP={best_params.clip} GAMMA={best_params.shrink_gamma} "
        f"| N_RECENT={n_recent} N_PACE={n_pace} PCAP={pcap} RIDGE={alpha} "
        f"| K_EARLY={k_early} K_LATE={k_late}"
    )

    raw_mae, cal_mae, out = final_test_on_s3(
        df_s1, df_s2, df_s3,
        best_params, n_recent, n_pace, pcap, alpha,
        k_early=k_early, k_late=k_late
    )

    print(f"\n{SEASON_3} Feb–Mar results:")
    print(f"TEST MAE raw Elo (pre-game):                 {raw_mae:.3f}")
    print(f"TEST MAE calibrated + venue-form + pace:     {cal_mae:.3f}")

    out.to_csv("data/processed/test_season3_febmar_with_preds.csv", index=False)
    print("Wrote: data/processed/test_season3_febmar_with_preds.csv")


if __name__ == "__main__":
    main()
