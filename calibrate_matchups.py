
import numpy as np
import pandas as pd

from backtest_matchups import (
    DATA_DIR,
    load_games,
    build_team_games,
    build_prior_multipliers,
    generate_predictions,
)

MARKETS = ["PTS", "REB"]
MODELS = ["BASELINE", "MATCHUP"]
BIN_CENTERS = np.arange(0.05, 1.0, 0.10)


def brier_score(y, p):
    return float(np.mean((p - y) ** 2))


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(
        -np.mean(
            y * np.log(p)
            + (1 - y) * np.log(1 - p)
        )
    )


def build_curve(raw, actual):
    bins = np.linspace(0, 1, 11)

    bin_ids = np.clip(
        np.digitize(raw, bins) - 1,
        0,
        9,
    )

    calibrated = []

    for i in range(10):
        mask = bin_ids == i
        count = int(mask.sum())

        observed = actual[mask].sum()

        baseline = (
            float(raw[mask].mean())
            if count
            else float(BIN_CENTERS[i])
        )

        adjusted = (
            observed + 100 * baseline
        ) / (count + 100)

        calibrated.append(adjusted)

    return np.maximum.accumulate(calibrated)


def main():
    print("Starting matchup probability calibration...")

    games = load_games()
    team_games = build_team_games(games)

    if team_games.empty:
        raise RuntimeError("No team-game data found.")

    multipliers = build_prior_multipliers(team_games)

    predictions = generate_predictions(
        games, multipliers
    )

    if predictions.empty:
        raise RuntimeError("No predictions generated.")

    # Use OVER only so each synthetic line
    # contributes one outcome per model.
    predictions = predictions[
        predictions["side"] == "OVER"
    ]

    results = []
    curves = []

    for market in MARKETS:
        for model in MODELS:
            subset = predictions[
                (predictions["market"] == market)
                & (predictions["model"] == model)
            ]

            train = subset[
                subset["dataset"] == "CALIBRATION"
            ]

            test = subset[
                subset["dataset"] == "EVALUATION"
            ]

            if len(train) < 100 or test.empty:
                raise RuntimeError(
                    f"Insufficient data: {market} {model}"
                )

            raw_train = train[
                "raw_probability"
            ].to_numpy(dtype=float)

            y_train = train[
                "won"
            ].to_numpy(dtype=float)

            raw_test = test[
                "raw_probability"
            ].to_numpy(dtype=float)

            y_test = test[
                "won"
            ].to_numpy(dtype=float)

            curve = build_curve(
                raw_train, y_train
            )

            calibrated = np.interp(
                raw_test,
                BIN_CENTERS,
                curve,
            )

            results.append({
                "market": market,
                "model": model,
                "predictions": len(test),
                "raw_brier": brier_score(
                    y_test, raw_test
                ),
                "calibrated_brier": brier_score(
                    y_test, calibrated
                ),
                "raw_log_loss": log_loss(
                    y_test, raw_test
                ),
                "calibrated_log_loss": log_loss(
                    y_test, calibrated
                ),
            })

            for p, cp in zip(BIN_CENTERS, curve):
                curves.append({
                    "market": market,
                    "model": model,
                    "raw_probability": float(p),
                    "calibrated_probability": float(cp),
                })

    report = pd.DataFrame(results)

    report["brier_improvement"] = (
        report["raw_brier"]
        - report["calibrated_brier"]
    )

    report["log_loss_improvement"] = (
        report["raw_log_loss"]
        - report["calibrated_log_loss"]
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        DATA_DIR / "matchup_calibration_report.csv",
        index=False,
    )

    pd.DataFrame(curves).to_csv(
        DATA_DIR / "matchup_calibration_curves.csv",
        index=False,
    )

    print("\nMATCHUP CALIBRATION RESULTS")
    print(report.to_string(index=False))

    print("\nSaved matchup calibration reports.")
    print("Synthetic lines only; no ROI inference.")


if __name__ == "__main__":
    main()
