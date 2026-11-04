
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
MARKETS = ["PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA"]


def brier_score(y, p):
    return np.mean((p - y) ** 2)


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(
        y * np.log(p) + (1 - y) * np.log(1 - p)
    )


def calibrate_probability(p, raw, actual):
    # Fit a smoothed probability adjustment using
    # calibration observations only.
    bins = np.linspace(0, 1, 11)

    bin_ids = np.clip(
        np.digitize(raw, bins) - 1, 0, 9
    )

    calibrated = []

    for i in range(10):
        mask = bin_ids == i
        count = mask.sum()

        # Shrink sparse bins toward the raw model.
        observed = actual[mask].sum()
        baseline = raw[mask].mean() if count else (i + 0.5) / 10

        adjusted = (
            observed + 100 * baseline
        ) / (count + 100)

        calibrated.append(adjusted)

    # Keep the calibration curve nondecreasing.
    calibrated = np.maximum.accumulate(calibrated)

    return np.interp(
        p, (bins[:-1] + bins[1:]) / 2,
        calibrated
    )


def main():
    calibration = pd.read_csv(
        DATA_DIR / "backtest_calibration.csv"
    )
    evaluation = pd.read_csv(
        DATA_DIR / "backtest_evaluation.csv"
    )

    results = []

    for market in MARKETS:
        train = calibration[
            (calibration["market"] == market)
            & (calibration["side"] == "OVER")
        ].copy()

        test = evaluation[
            (evaluation["market"] == market)
            & (evaluation["side"] == "OVER")
        ].copy()

        if len(train) < 100 or test.empty:
            continue

        raw_train = train["raw_probability"].to_numpy()
        y_train = train["won"].to_numpy()

        raw_test = test["raw_probability"].to_numpy()
        y_test = test["won"].to_numpy()

        adjusted = calibrate_probability(
            raw_test, raw_train, y_train
        )

        results.append({
            "market": market,
            "predictions": len(test),
            "raw_brier": brier_score(y_test, raw_test),
            "calibrated_brier": brier_score(y_test, adjusted),
            "raw_log_loss": log_loss(y_test, raw_test),
            "calibrated_log_loss": log_loss(y_test, adjusted),
        })

    report = pd.DataFrame(results)

    if report.empty:
        raise RuntimeError("No calibration results generated.")

    report["brier_improvement"] = (
        report["raw_brier"]
        - report["calibrated_brier"]
    )

    report["log_loss_improvement"] = (
        report["raw_log_loss"]
        - report["calibrated_log_loss"]
    )

    report.to_csv(
        DATA_DIR / "calibration_report.csv",
        index=False
    )

    print("\nNBA V4 CALIBRATION EVALUATION")
    print(report.to_string(index=False))

    print("\nPositive improvement means calibration helped.")
    print("Negative improvement means calibration made it worse.")


if __name__ == "__main__":
    main()
