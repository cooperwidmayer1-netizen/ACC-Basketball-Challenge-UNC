from __future__ import annotations

import pandas as pd

# ----------------------------
# SETTINGS
# ----------------------------
TRAIN_CUTOFF = pd.Timestamp("2026-02-07", tz="UTC")
PRED_START = TRAIN_CUTOFF
PRED_END = pd.Timestamp("2026-03-08", tz="UTC")  # end-exclusive

FETCH_START_ISO = "2025-11-01"
FETCH_END_ISO = "2026-03-08"
SLEEP_S = 0.15

PREDICT_FILTER = "acc_vs_acc"

# Pre-cutoff tuning split
VAL_START = pd.Timestamp("2026-01-10", tz="UTC")
VAL_END = TRAIN_CUTOFF

# ----------------------------
# BASIC MODEL
# ----------------------------
BASIC_ALPHA = 0.75

# ----------------------------
# TUNING GRIDS
# ----------------------------
WINSOR_GRID = [(0.19, 0.81), (0.21, 0.79), (0.23, 0.77)]
ALPHA_GRID = [0.72, 0.76, 0.80, 0.84]
FORM_W_GRID = [4, 5, 6, 7]

BLEND_W_GRID = [0.003, 0.005, 0.01, 0.03]
CLIP_GRID = [14, 16, 18, 20, 22]
SLOPE_SHRINK_GRID = [0.80, 0.85, 0.90]
FORM_SCALE_GRID = [1.0, 0.85, 0.70, 0.55]

# ----------------------------
# MANUAL NON-ACC GAMES
# ----------------------------
MANUAL_GAMES = [
    {"date": "2026-02-14", "away_team_short": "Baylor", "home_team_short": "Louisville"},
    {"date": "2026-02-14", "away_team_short": "Ohio State", "home_team_short": "Virginia"},
    {"date": "2026-02-21", "away_team_short": "Michigan", "home_team_short": "Duke"},
]
