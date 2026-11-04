
from pathlib import Path
import numpy as np
import pandas as pd
from math import erf, sqrt

DATA_DIR = Path("data")
SEASONS = [
    "2022-23", "2023-24", "2024-25",
    "2025-26", "2026-27"
]

MARKETS = ["PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA"]

# First 10 historical games are used for warmup.
MIN_HISTORY = 10
MAX_HISTORY = 20

# Date-based split: calibration data comes before evaluation.
EVALUATION_START = "2026-07-01"


def normal_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def get_market_values(df):
    df = df.copy()

    df["PTS"] = df["pts"]
    df["REB"] = df["reb"]
    df["AST"] = df["ast"]
    df["3PM"] = df["fg3m"]

    df["PRA"] = df["pts"] + df["reb"] + df["ast"]
    df["PR"] = df["pts"] + df["reb"]
    df["PA"] = df["pts"] + df["ast"]
    df["RA"] = df["reb"] + df["ast"]

    return df


def load_games():
    frames = []

    for season in SEASONS:
        path = DATA_DIR / f"game_logs_{season}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        df = pd.read_csv(path, low_memory=False)
        df["season"] = season
        frames.append(df)

    games = pd.concat(frames, ignore_index=True)

    games["game_date"] = pd.to_datetime(
        games["game_date"], errors="coerce"
    )

    numeric_cols = ["pts", "reb", "ast", "fg3m"]

    for col in numeric_cols:
        games[col] = pd.to_numeric(
            games[col], errors="coerce"
        )

    games = games.dropna(
        subset=["player_id", "game_id", "game_date"]
    )

    games = games.drop_duplicates(
        subset=["player_id", "game_id"]
    )

    games = games.sort_values(
        ["player_id", "game_date", "game_id"]
    ).reset_index(drop=True)

    return get_market_values(games)


def weighted_projection(history):
    l5 = history[-5:].mean()
    l10 = history[-10:].mean()
    l20 = history[-20:].mean()

    return 0.50 * l5 + 0.30 * l10 + 0.20 * l20


def main():
    games = load_games()
    results = []

    for player_id, group in games.groupby(
        "player_id", sort=False
    ):
        group = group.sort_values(
            ["game_date", "game_id"]
        ).reset_index(drop=True)

        for market in MARKETS:
            values = group[market].to_numpy(dtype=float)

            for i in range(MIN_HISTORY, len(group)):
                history = values[
                    max(0, i - MAX_HISTORY):i
                ]

                actual = values[i]

                if (
                    len(history) < MIN_HISTORY
                    or not np.isfinite(history).all()
                    or not np.isfinite(actual)
                ):
                    continue

                projection = weighted_projection(history)
                sd = np.std(history, ddof=1)

                if not np.isfinite(sd) or sd <= 0:
                    continue

               
                # Test five synthetic half-point lines.
                # These are NOT historical FanDuel lines.
                base_line = np.floor(projection) + 0.5
                offsets = [-4, -2, 0, 2, 4]

                game_date = group.loc[i, "game_date"]

                for offset in offsets:
                    line = base_line + offset

                    # Player statistics cannot be negative.
                    if line < 0.5:
                        continue

                    p_over = 1 - normal_cdf(
                        (line - projection) / sd
                    )
                    p_under = 1 - p_over

                    for side, probability, won in [
                        ("OVER", p_over, actual > line),
                        ("UNDER", p_under, actual < line),
                    ]:
                        results.append({
                            "game_date": game_date,
                            "player_id": player_id,
                            "market": market,
                            "side": side,
                            "line": line,
                            "projection": projection,
                            "sd": sd,
                            "raw_probability": probability,
                            "actual": actual,
                            "won": int(won),
                            "prediction_error": actual - projection,
                            "dataset": (
                                "EVALUATION"
                                if game_date >= pd.Timestamp(
                                    EVALUATION_START
                                )
                                else "CALIBRATION"
                            ),
                        })


    if not results:
        raise RuntimeError("No backtest predictions generated.")

   
   
    output = pd.DataFrame(results)

    output = output.sort_values(
        ["game_date", "player_id", "market", "side"]
    )

    # Separate historical calibration and future evaluation data
    calibration = output[
        output["dataset"] == "CALIBRATION"
    ].copy()

    evaluation = output[
        output["dataset"] == "EVALUATION"
    ].copy()

    # Keep up to 25,000 predictions from each period
  
    # Sample across each full date range, not just a few days.
    
    # Sample complete OVER/UNDER pairs across the date range.
    # Each pair represents one player-game-market-line.

    def sample_pairs(df, max_pairs=12500):
        over = df[df["side"] == "OVER"].copy()

        if over.empty:
            return df.iloc[0:0].copy()

        n = min(max_pairs, len(over))

        positions = np.linspace(
            0, len(over) - 1, n, dtype=int
        )

        selected = over.iloc[positions].copy()

        pair_keys = [
            "game_date", "player_id", "market", "line"
        ]

        paired = df.merge(
            selected[pair_keys].drop_duplicates(),
            on=pair_keys,
            how="inner"
        )

        return paired.sort_values(
            ["game_date", "player_id", "market", "side"]
        ).reset_index(drop=True)

    calibration = sample_pairs(calibration)
    evaluation = sample_pairs(evaluation)



    calibration_path = DATA_DIR / "backtest_calibration.csv"
    evaluation_path = DATA_DIR / "backtest_evaluation.csv"

    calibration.to_csv(calibration_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)

    print(f"Calibration predictions: {len(calibration):,}")
    print(f"Evaluation predictions: {len(evaluation):,}")

    print(
        "Calibration dates:",
        calibration["game_date"].min(),
        "to",
        calibration["game_date"].max()
    )

    print(
        "Evaluation dates:",
        evaluation["game_date"].min(),
        "to",
        evaluation["game_date"].max()
    )



   
    
    # Evaluate OVER predictions by market.
    for name, df in [
        ("CALIBRATION", calibration),
        ("EVALUATION", evaluation),
    ]:
        over = df[df["side"] == "OVER"].copy()

        over["brier"] = (
            over["raw_probability"] - over["won"]
        ) ** 2

        summary = over.groupby("market").agg(
            predictions=("won", "size"),
            predicted_probability=("raw_probability", "mean"),
            actual_hit_rate=("won", "mean"),
            brier_score=("brier", "mean"),
        )

        print(f"\n{name} OVER BACKTEST")
        print(summary.to_string())



    # Probability accuracy report using evaluation data
    print("\nEVALUATION PROBABILITY ACCURACY")

    eval_over = evaluation[
        evaluation["side"] == "OVER"
    ].copy()

    bins = np.arange(0, 1.01, 0.10)

    eval_over["probability_range"] = pd.cut(
        eval_over["raw_probability"],
        bins=bins,
        include_lowest=True
    )

    accuracy = eval_over.groupby(
        "probability_range", observed=True
    ).agg(
        predictions=("won", "size"),
        avg_predicted=("raw_probability", "mean"),
        actual_hit_rate=("won", "mean")
    )

    accuracy["difference"] = (
        accuracy["avg_predicted"]
        - accuracy["actual_hit_rate"]
    )

    print(accuracy.to_string())

if __name__ == "__main__":
    main()
