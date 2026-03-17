from __future__ import annotations

import pandas as pd

# Goal: make ESPN names and CSV names match.
NAME_FIX = {
    "NC State": "North Carolina State",
    "N.C. State": "North Carolina State",
    "Miami": "Miami (FL)",
    "Miami FL": "Miami (FL)",
    "Pitt": "Pittsburgh",
    "Cal": "California",
    "Florida St": "Florida State",
    "Florida St.": "Florida State",
    "UNC": "North Carolina",
}

ACC_TEAMS = {
    "Boston College",
    "California",
    "Clemson",
    "Duke",
    "Florida State",
    "Georgia Tech",
    "Louisville",
    "Miami (FL)",
    "North Carolina State",
    "North Carolina",
    "Notre Dame",
    "Pittsburgh",
    "SMU",
    "Stanford",
    "Syracuse",
    "Virginia",
    "Virginia Tech",
    "Wake Forest",
}


def norm_team(x: str) -> str:
    """Normalize team names across ESPN and the CSV authority schedule."""
    if pd.isna(x):
        return x
    s = str(x).strip()
    s = " ".join(s.split())
    s = s.replace("&", "and")
    return NAME_FIX.get(s, s)
