import pandas as pd

def load_authority_schedule(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df

def merge_schedule(fetched, authority):
    keep = ["game_id", "date", "home", "away"]
    authority = authority[keep]

    merged = authority.merge(
        fetched,
        on=["game_id", "date", "home", "away"],
        how="left"
    )
    return merged
