
import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
INPUT_FILES = [
    "backtest_calibration.csv",
    "backtest_evaluation.csv",
]

OUTPUT_FILE = os.path.join(DATA_DIR, "walkforward_results.csv")
CURVES_FILE = os.path.join(DATA_DIR, "walkforward_curves.csv")

MIN_TRAIN = 100
SHRINKAGE = 100
BINS = np.linspace(0, 1, 11)
CENTERS = (BINS[:-1] + BINS[1:]) / 2

def brier(y, p):
    return np.mean((p - y) ** 2)

def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

def train_curve(df):
    p = df["raw_probability"].to_numpy()
    y = df["hit"].to_numpy()

    calibrated = []

    for i in range(10):
        if i == 9:
            mask = (p >= BINS[i]) & (p <= BINS[i+1])
        else:
            mask = (p >= BINS[i]) & (p < BINS[i+1])

        n = mask.sum()
        hits = y[mask].sum()

        value = (hits + SHRINKAGE * CENTERS[i]) / (
            n + SHRINKAGE
        )

        calibrated.append(value)

    return np.maximum.accumulate(calibrated)

def main():
    frames = []

    for filename in INPUT_FILES:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("No backtest files found.")

    df = pd.concat(frames, ignore_index=True)
    print("CSV COLUMN NAMES:", df.columns.tolist(), flush=True)
    required = [
        "date",
        "market",
        "side",
        "raw_probability",
        "hit",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=required)

    df = df[df["side"].str.upper() == "OVER"].copy()
    df = df.sort_values("date")

    df["season_start"] = np.where(
        df["date"].dt.month >= 7,
        df["date"].dt.year,
        df["date"].dt.year - 1,
    )

    results = []
    curve_rows = []

    for test_season in sorted(df["season_start"].unique()):
        train = df[df["season_start"] < test_season]
        test = df[df["season_start"] == test_season]

        if train.empty or test.empty:
            continue

        for market in sorted(test["market"].unique()):
            tr = train[train["market"] == market]
            te = test[test["market"] == market]

            if len(tr) < MIN_TRAIN or te.empty:
                continue

            curve = train_curve(tr)

            raw = te["raw_probability"].to_numpy()
            actual = te["hit"].to_numpy()

            calibrated = np.interp(
                raw,
                CENTERS,
                curve,
            )

            results.append({
                "test_season": test_season,
                "market": market,
                "train_rows": len(tr),
                "test_rows": len(te),
                "raw_brier": brier(actual, raw),
                "cal_brier": brier(actual, calibrated),
                "raw_logloss": logloss(actual, raw),
                "cal_logloss": logloss(actual, calibrated),
            })

            for x, y in zip(CENTERS, curve):
                curve_rows.append({
                    "test_season": test_season,
                    "market": market,
                    "raw_probability": x,
                    "calibrated_probability": y,
                })

    os.makedirs(DATA_DIR, exist_ok=True)

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE, index=False
    )

    pd.DataFrame(curve_rows).to_csv(
        CURVES_FILE, index=False
    )

    print("Walk-forward validation complete.")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
