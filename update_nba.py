
from pathlib import Path
import pandas as pd

BASE_URL = (
    "https://huggingface.co/datasets/"
    "cbratkovics/nba-game-logs/resolve/main/game_logs/"
)

SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

for season in SEASONS:
    print(f"Downloading {season}...")

    url = BASE_URL + f"game_logs_{season}.parquet"
    df = pd.read_parquet(url)

    df = df.drop_duplicates(
        subset=["player_id", "game_id"]
    )

    df["game_date"] = pd.to_datetime(
        df["game_date"]
    ).dt.strftime("%Y-%m-%d")

    output_file = (
        output_dir / f"game_logs_{season}.csv"
    )

    df.to_csv(output_file, index=False)

    print(
        f"{season}: {len(df)} records, "
        f"latest game: {df['game_date'].max()}"
    )

# Generate current-season team-game results.
season = "2026-27"

player_file = output_dir / f"game_logs_{season}.csv"
players = pd.read_csv(player_file, low_memory=False)

required = [
    "game_id", "game_date", "team", "opponent",
    "pts", "reb", "ast", "fg3m"
]

missing = [c for c in required if c not in players.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in ["pts", "reb", "ast", "fg3m"]:
    players[col] = pd.to_numeric(
        players[col], errors="coerce"
    )

players["team"] = players["team"].astype(str).str.strip()
players["opponent"] = players["opponent"].astype(str).str.strip()

team_games = (
    players.groupby(
        ["game_id", "game_date", "team", "opponent"],
        as_index=False
    )
    .agg(
        points=("pts", "sum"),
        rebounds=("reb", "sum"),
        assists=("ast", "sum"),
        three_pointers=("fg3m", "sum"),
        player_count=("player_id", "nunique")
    )
)

opposing = team_games[
    ["game_id", "team", "points"]
].rename(columns={
    "team": "opponent",
    "points": "opponent_points"
})

team_games = team_games.merge(
    opposing,
    on=["game_id", "opponent"],
    how="left",
    validate="many_to_one"
)

# Retain games where the opponent's team total exists.
team_games = team_games.dropna(
    subset=["opponent_points"]
).copy()

team_games["point_diff"] = (
    team_games["points"] - team_games["opponent_points"]
)

team_games["season"] = season

team_games = team_games.sort_values(
    ["game_date", "game_id", "team"]
)

output_file = output_dir / "team_game_results_2026-27.csv"
team_games.to_csv(output_file, index=False)

print(
    f"Team results: {len(team_games)} records, "
    f"latest game: {team_games['game_date'].max()}"
)

print("NBA data conversion completed!")
