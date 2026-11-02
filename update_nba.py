
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

print("NBA data conversion completed!")
