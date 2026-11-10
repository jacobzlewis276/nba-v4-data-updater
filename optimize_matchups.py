
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_matchups import (
    DATA_DIR,
    EVALUATION_START,
    MAX_ADJUSTMENT,
    build_team_games,
    build_prior_multipliers,
    generate_predictions,
    load_games,
)

STRENGTHS = [0.00, 0.10, 0.20, 0.30, 0.40]


def main():
    print("Starting matchup shrinkage optimization...")

    games = load_games()
    team_games = build_team_games(games)

    if team_games.empty:
        raise RuntimeError("No paired team games found.")

    # Existing backtest computes historical multipliers
    # using 20% shrinkage and the 8% cap.
    multipliers = build_prior_multipliers(team_games)

    # Reconstruct the uncapped defensive ratios.
    # This is only valid where the original multiplier
    # did not hit the 8% cap.
    #
    # To avoid relying on that approximation, this
    # optimization tests strengths from the original
    # ratio whenever the 20% multiplier is uncapped.

    records = []

    for strength in STRENGTHS:
        test_multipliers = multipliers.copy()

        for market in ["PTS", "REB"]:
            column = f"{market}_multiplier"

            original = test_multipliers[column]

            ratio = 1 + (original - 1) / 0.20

            adjusted = (
                1 + strength * (ratio - 1)
            ).clip(
                1 - MAX_ADJUSTMENT,
                1 + MAX_ADJUSTMENT,
            )

            test_multipliers[column] = adjusted

        predictions = generate_predictions(
            games, test_multipliers
        )

        # The MATCHUP rows contain the projection
        # corresponding to this tested strength.
        selected = predictions[
            (predictions["model"] == "MATCHUP")
            & (predictions["side"] == "OVER")
        ].copy()

        summary = (
            selected.groupby(
                ["dataset", "market"],
                as_index=False,
            )
            .agg(
                predictions=("won", "size"),
                mae=("absolute_error", "mean"),
                rmse=(
                    "squared_error",
                    lambda x: np.sqrt(x.mean()),
                ),
                brier=("brier", "mean"),
            )
        )

        summary["strength"] = strength
        records.append(summary)

        print(f"Finished strength {strength:.2f}")

    results = pd.concat(
        records, ignore_index=True
    )

    # Select strength using calibration data ONLY.
    calibration = results[
        results["dataset"] == "CALIBRATION"
    ].copy()

    best = (
        calibration.sort_values(
            ["market", "mae", "strength"]
        )
        .groupby("market", as_index=False)
        .first()[["market", "strength"]]
    )

    best = best.rename(
        columns={"strength": "selected_strength"}
    )

    evaluation = results[
        results["dataset"] == "EVALUATION"
    ].merge(
        best, on="market", how="left"
    )

    selected_evaluation = evaluation[
        np.isclose(
            evaluation["strength"],
            evaluation["selected_strength"],
        )
    ].copy()

    DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    results.to_csv(
        DATA_DIR / "matchup_strength_results.csv",
        index=False,
    )

    selected_evaluation.to_csv(
        DATA_DIR / "matchup_strength_selected.csv",
        index=False,
    )

    print()
    print("CALIBRATION-SELECTED STRENGTHS")
    print(best.to_string(index=False))

    print()
    print("HELD-OUT EVALUATION RESULTS")
    print(selected_evaluation.to_string(index=False))

    print()
    print("Saved matchup optimization CSV files.")
    print("No existing backtest outputs were changed.")


if __name__ == "__main__":
    main()
