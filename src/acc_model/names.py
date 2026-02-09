TEAM_REMAP = {
    "Miami (FL)": "Miami",
    "Florida St.": "Florida State",
    "N.C. State": "NC State",
    "UNC": "North Carolina",
}

def normalize_team(name: str) -> str:
    return TEAM_REMAP.get(name, name)
