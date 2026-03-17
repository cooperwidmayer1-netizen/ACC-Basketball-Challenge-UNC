from __future__ import annotations

import numpy as np
import pandas as pd

from .espn import make_game_id
from .names import norm_team


def load_csv_authority_schedule(
    csv_path: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load the authority CSV schedule and keep only rows in [start_ts, end_ts).
    Expects columns:
      Date, Visitor/Neutral, Home/Neutral
    """
    s = pd.read_csv(csv_path)

    expected = {"Date", "Visitor/Neutral", "Home/Neutral"}
    if not expected.issubset(s.columns):
        raise ValueError(f"CSV missing expected columns. Found: {list(s.columns)}")

    s["date"] = pd.to_datetime(s["Date"], utc=True, errors="coerce")
    s["away_team_short"] = s["Visitor/Neutral"].apply(norm_team)
    s["home_team_short"] = s["Home/Neutral"].apply(norm_team)

    s = s.dropna(subset=["date", "home_team_short", "away_team_short"]).copy()
    s = s[(s["date"] >= start_ts) & (s["date"] < end_ts)].copy()

    # Neutral-site info is not supplied in the CSV, so follow the notebook.
    s["neutral_site"] = False
    s["game_id"] = make_game_id(s)
    s["date_key"] = s["date"].dt.date

    keep = ["game_id", "date", "date_key", "home_team_short", "away_team_short", "neutral_site"]
    return s[keep].reset_index(drop=True)


def merge_csv_with_espn_future(csv_future: pd.DataFrame, espn_future: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all CSV schedule games and enrich with ESPN identifiers / neutral-site values when possible.
    """
    e = espn_future.copy()
    if "date_key" not in e.columns:
        e["date_key"] = e["date"].dt.date

    merged = csv_future.merge(
        e[["game_id", "date_key", "home_team_short", "away_team_short", "neutral_site"]],
        on=["date_key", "home_team_short", "away_team_short"],
        how="left",
        suffixes=("_csv", "_espn"),
        indicator=True,
    )

    gid_csv = "game_id_csv" if "game_id_csv" in merged.columns else "game_id"
    gid_espn = "game_id_espn" if "game_id_espn" in merged.columns else None
    ns_csv = "neutral_site_csv" if "neutral_site_csv" in merged.columns else "neutral_site"
    ns_espn = "neutral_site_espn" if "neutral_site_espn" in merged.columns else None

    if gid_espn is not None:
        merged["game_id_final"] = merged[gid_espn].fillna(merged[gid_csv])
    else:
        merged["game_id_final"] = merged[gid_csv]

    if ns_espn is not None:
        merged["neutral_site_final"] = merged[ns_espn].fillna(merged[ns_csv]).astype(bool)
    else:
        merged["neutral_site_final"] = merged[ns_csv].astype(bool)

    missing = int((merged["_merge"] == "left_only").sum())
    print(f"CSV schedule games (authority): {len(csv_future)}")
    print(f"ESPN future games (acc_vs_acc): {len(espn_future)}")
    print(f"Missing from ESPN but kept from CSV: {missing}")

    out = pd.DataFrame(
        {
            "game_id": merged["game_id_final"],
            "date": merged["date"],
            "home_team_short": merged["home_team_short"],
            "away_team_short": merged["away_team_short"],
            "neutral_site": merged["neutral_site_final"],
        }
    ).sort_values(["date", "game_id"]).reset_index(drop=True)

    print(f"FINAL future games to predict: {len(out)}")
    return out


def build_rest_from_schedule(df_sched: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest-day features from schedule only, matching the notebook.
    """
    d = df_sched.sort_values(["date", "game_id"]).reset_index(drop=True).copy()

    home_long = d[["game_id", "date", "home_team_short"]].rename(columns={"home_team_short": "team"})
    away_long = d[["game_id", "date", "away_team_short"]].rename(columns={"away_team_short": "team"})
    long = (
        pd.concat([home_long, away_long], ignore_index=True)
        .sort_values(["team", "date", "game_id"])
        .reset_index(drop=True)
    )

    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.total_seconds() / 86400.0
    long["rest_days"] = long["rest_days"].clip(lower=0, upper=14)

    rest_fallback = float(np.nanmedian(long["rest_days"]))
    long["rest_days"] = long["rest_days"].fillna(rest_fallback)

    home_map = long.merge(
        d[["game_id", "home_team_short"]].rename(columns={"home_team_short": "team"}),
        on=["game_id", "team"],
        how="right",
    )[["game_id", "rest_days"]].rename(columns={"rest_days": "home_rest_days"})

    away_map = long.merge(
        d[["game_id", "away_team_short"]].rename(columns={"away_team_short": "team"}),
        on=["game_id", "team"],
        how="right",
    )[["game_id", "rest_days"]].rename(columns={"rest_days": "away_rest_days"})

    m = (
        d[["game_id"]]
        .drop_duplicates("game_id")
        .merge(home_map, on="game_id", how="left")
        .merge(away_map, on="game_id", how="left")
    )

    m["rest_diff"] = m["home_rest_days"] - m["away_rest_days"]
    for c in ["home_rest_days", "away_rest_days", "rest_diff"]:
        m[c] = m[c].fillna(0.0)

    return m
