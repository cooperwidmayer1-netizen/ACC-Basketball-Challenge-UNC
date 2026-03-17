from __future__ import annotations

import numpy as np
import pandas as pd


def add_no_leak_form_tempo(
    games: pd.DataFrame,
    scored_games: pd.DataFrame,
    form_w: int,
    min_p: int = 2,
) -> pd.DataFrame:
    """
    Add no-leak rolling form and tempo estimates as-of each game.
    This follows the notebook exactly: rolling mean shifted by one game.
    """
    g = games.copy()

    sg = scored_games.dropna(subset=["home_score", "away_score"]).copy()
    if "game_id" not in sg.columns:
        raise ValueError(
            "scored_games is missing 'game_id' — ensure dedup() was run before creating df_scored."
        )

    sg = sg.sort_values(["date", "game_id"]).reset_index(drop=True)

    home_long = sg[["game_id", "date", "home_team_short", "home_score", "away_score"]].rename(
        columns={"home_team_short": "team"}
    )
    home_long["pts_for"] = home_long["home_score"]
    home_long["pts_against"] = home_long["away_score"]

    away_long = sg[["game_id", "date", "away_team_short", "home_score", "away_score"]].rename(
        columns={"away_team_short": "team"}
    )
    away_long["pts_for"] = away_long["away_score"]
    away_long["pts_against"] = away_long["home_score"]

    long = pd.concat([home_long, away_long], ignore_index=True)
    long["margin_team"] = long["pts_for"] - long["pts_against"]
    long["tempo_game"] = (long["pts_for"] + long["pts_against"]) / 2.0
    long = long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    long["form_margin"] = (
        long.groupby("team")["margin_team"]
        .apply(lambda s: s.rolling(form_w, min_periods=min_p).mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    long["tempo_hat"] = (
        long.groupby("team")["tempo_game"]
        .apply(lambda s: s.rolling(form_w, min_periods=min_p).mean().shift(1))
        .reset_index(level=0, drop=True)
    )

    form_fb = float(np.nanmean(long["margin_team"])) if len(long) else 0.0
    tempo_fb = float(np.nanmean(long["tempo_game"])) if len(long) else 70.0
    long["form_margin"] = long["form_margin"].fillna(form_fb)
    long["tempo_hat"] = long["tempo_hat"].fillna(tempo_fb)

    home_map = long[["game_id", "team", "form_margin", "tempo_hat"]].rename(
        columns={
            "team": "home_team_short",
            "form_margin": "home_form_margin",
            "tempo_hat": "home_tempo_hat",
        }
    )
    away_map = long[["game_id", "team", "form_margin", "tempo_hat"]].rename(
        columns={
            "team": "away_team_short",
            "form_margin": "away_form_margin",
            "tempo_hat": "away_tempo_hat",
        }
    )

    g = g.merge(home_map, on=["game_id", "home_team_short"], how="left")
    g = g.merge(away_map, on=["game_id", "away_team_short"], how="left")

    g["home_form_margin"] = g["home_form_margin"].fillna(form_fb).astype(float)
    g["away_form_margin"] = g["away_form_margin"].fillna(form_fb).astype(float)
    g["form_diff"] = g["home_form_margin"] - g["away_form_margin"]

    g["home_tempo_hat"] = g["home_tempo_hat"].fillna(tempo_fb).astype(float)
    g["away_tempo_hat"] = g["away_tempo_hat"].fillna(tempo_fb).astype(float)
    g["tempo"] = (g["home_tempo_hat"] + g["away_tempo_hat"]) / 2.0

    return g


def freeze_team_form_tempo_at_cutoff(
    scored_games: pd.DataFrame,
    cutoff: pd.Timestamp,
    form_w: int,
    min_p: int = 2,
):
    """
    Freeze each team's most recent pre-cutoff form and tempo for future prediction.
    """
    sg = scored_games.dropna(subset=["home_score", "away_score"]).copy()
    if "game_id" not in sg.columns:
        raise ValueError(
            "scored_games is missing 'game_id' — ensure dedup() was run before creating df_scored."
        )

    sg = sg[sg["date"] < cutoff].sort_values(["date", "game_id"]).reset_index(drop=True)

    home_long = sg[["game_id", "date", "home_team_short", "home_score", "away_score"]].rename(
        columns={"home_team_short": "team"}
    )
    home_long["pts_for"] = home_long["home_score"]
    home_long["pts_against"] = home_long["away_score"]

    away_long = sg[["game_id", "date", "away_team_short", "home_score", "away_score"]].rename(
        columns={"away_team_short": "team"}
    )
    away_long["pts_for"] = away_long["away_score"]
    away_long["pts_against"] = away_long["home_score"]

    long = pd.concat([home_long, away_long], ignore_index=True)
    long["margin_team"] = long["pts_for"] - long["pts_against"]
    long["tempo_game"] = (long["pts_for"] + long["pts_against"]) / 2.0
    long = long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    long["form_margin"] = (
        long.groupby("team")["margin_team"]
        .apply(lambda s: s.rolling(form_w, min_periods=min_p).mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    long["tempo_hat"] = (
        long.groupby("team")["tempo_game"]
        .apply(lambda s: s.rolling(form_w, min_periods=min_p).mean().shift(1))
        .reset_index(level=0, drop=True)
    )

    form_fb = float(np.nanmean(long["margin_team"])) if len(long) else 0.0
    tempo_fb = float(np.nanmean(long["tempo_game"])) if len(long) else 70.0
    long["form_margin"] = long["form_margin"].fillna(form_fb)
    long["tempo_hat"] = long["tempo_hat"].fillna(tempo_fb)

    last = long.groupby("team").tail(1)
    team_form = dict(zip(last["team"], last["form_margin"]))
    team_tempo = dict(zip(last["team"], last["tempo_hat"]))

    return team_form, team_tempo, form_fb, tempo_fb


def attach_frozen_form_tempo_future(
    df_games: pd.DataFrame,
    team_form: dict,
    team_tempo: dict,
    form_fb: float,
    tempo_fb: float,
    form_scale: float = 1.0,
) -> pd.DataFrame:
    """Attach frozen cutoff features to future games."""
    d = df_games.copy()
    d["home_form_margin"] = (
        d["home_team_short"].map(team_form).fillna(form_fb).astype(float) * float(form_scale)
    )
    d["away_form_margin"] = (
        d["away_team_short"].map(team_form).fillna(form_fb).astype(float) * float(form_scale)
    )
    d["form_diff"] = d["home_form_margin"] - d["away_form_margin"]

    d["tempo"] = (
        d["home_team_short"].map(team_tempo).fillna(tempo_fb).astype(float)
        + d["away_team_short"].map(team_tempo).fillna(tempo_fb).astype(float)
    ) / 2.0

    return d
