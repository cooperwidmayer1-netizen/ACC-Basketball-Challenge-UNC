import json
import time
import urllib.request
import pandas as pd
from .names import normalize_team

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

def fetch_espn(start_iso, end_iso, sleep_s=0.15):
    rows = []
    day = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)

    while day <= end:
        url = f"{BASE}?dates={day.strftime('%Y%m%d')}"
        with urllib.request.urlopen(url) as r:
            js = json.loads(r.read())
        time.sleep(sleep_s)

        for ev in js.get("events", []):
            comp = ev["competitions"][0]
            teams = comp["competitors"]

            home = next(t for t in teams if t["homeAway"] == "home")
            away = next(t for t in teams if t["homeAway"] == "away")

            rows.append({
                "game_id": ev["id"],
                "date": pd.to_datetime(comp["date"], utc=True),
                "home": normalize_team(home["team"]["displayName"]),
                "away": normalize_team(away["team"]["displayName"]),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "status": comp["status"]["type"]["name"]
            })
        day += pd.Timedelta(days=1)

    return pd.DataFrame(rows)
