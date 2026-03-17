from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date

import pandas as pd

from .names import norm_team

BASE_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)


def get_json(url: str, params: dict | None = None) -> dict:
    """Fetch JSON from an ESPN endpoint."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def month_ranges(start_iso: str, end_iso: str):
    """
    Yield ESPN-compatible date ranges month by month, e.g. YYYYMMDD-YYYYMMDD.
    This mirrors the notebook and avoids brittle daily fetching.
    """
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    cur = date(start.year, start.month, 1)

    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), 1 if cur.month == 12 else cur.month + 1, 1)
        month_end = (pd.Timestamp(nxt) - pd.Timedelta(days=1)).date()
        s = max(cur, start)
        e = min(month_end, end)
        yield f"{s:%Y%m%d}-{e:%Y%m%d}"
        cur = nxt


def extract_games(payload: dict) -> list[dict]:
    """
    Extract games from ESPN payload while preserving the notebook logic:
    - skip season type 1
    - for season type 3, keep only conferenceCompetition games
    - keep neutral-site flag
    - use short display names when available
    """
    rows: list[dict] = []

    for event in payload.get("events", []):
        season_type = (event.get("season") or {}).get("type")
        if season_type in (1, "1"):
            continue

        for comp in event.get("competitions", []):
            if season_type in (3, "3") and not comp.get("conferenceCompetition"):
                continue

            competitors = comp.get("competitors", []) or []
            if len(competitors) != 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            home_team = home.get("team") or {}
            away_team = away.get("team") or {}

            rows.append(
                {
                    "date": event.get("date"),
                    "neutral_site": bool(comp.get("neutralSite")),
                    "home_team_short": home_team.get("shortDisplayName") or home_team.get("displayName"),
                    "away_team_short": away_team.get("shortDisplayName") or away_team.get("displayName"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                }
            )
    return rows


def fetch_season(start_iso: str, end_iso: str, sleep_s: float = 0.2) -> list[dict]:
    """Fetch the season in monthly chunks for both regular season and postseason."""
    rows: list[dict] = []

    for dr in month_ranges(start_iso, end_iso):
        payload = get_json(
            BASE_SCOREBOARD,
            params={"dates": dr, "groups": 50, "seasontype": 2, "limit": 1000},
        )
        rows.extend(extract_games(payload))
        time.sleep(sleep_s)

        payload = get_json(
            BASE_SCOREBOARD,
            params={"dates": dr, "groups": 50, "seasontype": 3, "limit": 1000},
        )
        rows.extend(extract_games(payload))
        time.sleep(sleep_s)

    return rows


def make_game_id(d: pd.DataFrame) -> pd.Series:
    """Create the notebook-stable game id."""
    day = pd.to_datetime(d["date"], utc=True).dt.strftime("%Y%m%d")
    return (
        day
        + "|"
        + d["home_team_short"].astype(str)
        + "|"
        + d["away_team_short"].astype(str)
        + "|"
        + d["neutral_site"].astype(int).astype(str)
    )


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate ESPN rows exactly the way the notebook does:
    first by raw identifying fields, then by stable game_id.
    """
    out = df.sort_values("date").copy()
    out = out.drop_duplicates(
        subset=["date", "home_team_short", "away_team_short", "neutral_site"],
        keep="first",
    )
    out["game_id"] = make_game_id(out)
    out = out.sort_values(["date", "game_id"]).drop_duplicates("game_id", keep="first")
    return out.reset_index(drop=True)


def fetch_and_prepare_games(start_iso: str, end_iso: str, sleep_s: float) -> pd.DataFrame:
    """Notebook-faithful fetch + clean + normalize + dedup pipeline."""
    df = pd.DataFrame(fetch_season(start_iso, end_iso, sleep_s=sleep_s))
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_team_short", "away_team_short"]).copy()

    df["home_team_short"] = df["home_team_short"].apply(norm_team)
    df["away_team_short"] = df["away_team_short"].apply(norm_team)

    df = dedup(df)
    return df
