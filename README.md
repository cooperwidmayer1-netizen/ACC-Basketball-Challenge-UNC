# ACC Basketball Challenge

This project builds a predictive pipeline for ACC basketball games using historical results, schedule data, team-level rolling features, and ridge regression models.
## Overview

The model combines historical game data from ESPN with a schedule CSV containing upcoming games. It then constructs team-based features such as recent form, tempo, and rest before fitting models to estimate future game spreads.

The project is designed to:
- fetch and prepare historical game data
- normalize team names across data sources
- engineer no-leak rolling features
- train predictive models on past games
- tune model settings on a validation window
- generate predictions for future games

## Project Structure

```text
ACC-Basketball-Challenge/
├── main.py
├── README.md
├── requirements.txt
├── def2.ipynb
├── data/
│   └── raw/
│       └── acc_2025_2026.csv
└── src/
    └── acc_model/
        ├── __init__.py
        ├── config.py
        ├── espn.py
        ├── features.py
        ├── models.py
        ├── names.py
        ├── pipeline.py
        └── schedule.py
