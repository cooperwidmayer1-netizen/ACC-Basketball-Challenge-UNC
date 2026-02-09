import pandas as pd

# =====================
# DATE CUTS (NO LEAK)
# =====================
TRAIN_CUTOFF = pd.Timestamp("2026-02-07", tz="UTC")

FETCH_START_ISO = "2025-11-01"
FETCH_END_ISO   = "2026-03-08"

# =====================
# MODEL SETTINGS
# =====================
ROLL_FORM = 5
ROLL_TEMPO = 10
RIDGE_ALPHA = 1.0

MAX_MARGIN_CAP = 15
BLOWOUT_START = 15
BLOWOUT_WEIGHT = 0.5

# =====================
# GRID SEARCH
# =====================
ALPHAS = [0.1, 0.5, 1.0, 2.0]

# =====================
# REST DAYS
# =====================
MAX_REST_DAYS = 7