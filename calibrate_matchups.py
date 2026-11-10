
from pathlib import Path
from math import erf, sqrt

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]

MARKETS = {
    "PTS": "pts",
    "REB": "reb",
}

MIN_HISTORY = 10
MAX_HISTORY = 20

SHRINKAGE = 0.20
MAX_ADJUSTMENT = 0.08

EVALUATION_START = pd.Timestamp("2026-07-01")


def normal_cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


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
    ).dt.normalize()

    for col in ["pts", "reb"]:
        games[col] = pd.to_numeric(
            games[col], errors="coerce"
        )

    games = games.dropna(
        subset=[
            "game_id",
            "game_date",
            "player_id",
            "team",
            "opponent",
        ]
    )

    games = games.drop_duplicates(
        subset=["player_id", "game_id"]
    )

    for col in ["team", "opponent"]:
        games[col] = (
            games[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"SAS": "SAN"})
        )

    return games.sort_values(
        ["game_date", "game_id", "player_id"]
    ).reset_index(drop=True)


def build_team_games(games):
    # Combine player statistics into team-game totals.

    team_games = (
        games.groupby(
            ["game_id", "game_date", "team", "opponent"],
            as_index=False,
        )
        .agg(
            pts=("pts", "sum"),
            reb=("reb", "sum"),
        )
    )

    # Match each team with its opponent in the same game.

    paired = team_games.merge(
        team_games[
            ["game_id", "team", "pts", "reb"]
        ].rename(
            columns={
                "team": "opponent",
                "pts": "opp_pts",
                "reb": "opp_reb",
            }
        ),
        on=["game_id", "opponent"],
        how="inner",
        validate="one_to_one",
    )

    paired = paired[
        paired["team"] != paired["opponent"]
    ].copy()

    return paired.sort_values(
        ["game_date", "game_id", "team"]
    ).reset_index(drop=True)



def build_prior_multipliers(team_games):
    # Reset defensive averages each NBA season.
    # Use only games from BEFORE the prediction date.

    tg = team_games.copy()

    # NBA seasons begin in the fall.
    # July through December belong to the season
    # beginning that calendar year.
    year = tg["game_date"].dt.year
    month = tg["game_date"].dt.month

    tg["season_start"] = np.where(
        month >= 7, year, year - 1
    )

    daily = (
        tg.groupby(
            ["season_start", "game_date", "team"],
            as_index=False,
        )
        .agg(
            opp_pts=("opp_pts", "sum"),
            opp_reb=("opp_reb", "sum"),
            games=("game_id", "nunique"),
        )
    )

    league_daily = (
        daily.groupby(
            ["season_start", "game_date"],
            as_index=False,
        )
        .agg(
            league_pts=("opp_pts", "sum"),
            league_reb=("opp_reb", "sum"),
            league_games=("games", "sum"),
        )
        .sort_values(
            ["season_start", "game_date"]
        )
    )

    # League totals restart at each season.

    for col in ["pts", "reb", "games"]:
        source = (
            "league_games"
            if col == "games"
            else f"league_{col}"
        )

        cumulative = league_daily.groupby(
            "season_start"
        )[source].cumsum()

        league_daily[f"prior_{col}"] = (
            cumulative.groupby(
                league_daily["season_start"]
            ).shift(fill_value=0)
        )

    dates = league_daily[
        [
            "season_start",
            "game_date",
            "prior_pts",
            "prior_reb",
            "prior_games",
        ]
    ].copy()

    teams = daily.sort_values(
        ["season_start", "team", "game_date"]
    ).copy()

    # Team defensive totals restart each season.

    for col in ["opp_pts", "opp_reb", "games"]:
        teams[f"cumulative_{col}"] = (
            teams.groupby(
                ["season_start", "team"]
            )[col].cumsum()
        )

    team_frames = []

    for season_start, season_dates in dates.groupby(
        "season_start", sort=False
    ):
        season_teams = sorted(
            teams.loc[
                teams["season_start"] == season_start,
                "team",
            ].unique()
        )

        idx = pd.DatetimeIndex(
            season_dates["game_date"]
        )

        for team in season_teams:
            subset = teams[
                (teams["season_start"] == season_start)
                & (teams["team"] == team)
            ].set_index("game_date")

            cumulative = subset[
                [
                    "cumulative_opp_pts",
                    "cumulative_opp_reb",
                    "cumulative_games",
                ]
            ]

            # Shift one calendar date to avoid
            # using any same-day results.

            cumulative = (
                cumulative.reindex(idx)
                .ffill()
                .shift(1)
                .fillna(0)
            )

            cumulative = cumulative.rename(
                columns={
                    "cumulative_opp_pts": "team_prior_pts",
                    "cumulative_opp_reb": "team_prior_reb",
                    "cumulative_games": "team_prior_games",
                }
            )

            cumulative["game_date"] = idx.to_numpy()
            cumulative["season_start"] = season_start
            cumulative["opponent"] = team

            team_frames.append(
                cumulative.reset_index(drop=True)
            )

    prior = pd.concat(
        team_frames, ignore_index=True
    ).merge(
        dates,
        on=["season_start", "game_date"],
        how="left",
    )

    for market, team_col, league_col in [
        ("PTS", "team_prior_pts", "prior_pts"),
        ("REB", "team_prior_reb", "prior_reb"),
    ]:
        team_avg = (
            prior[team_col]
            / prior["team_prior_games"].replace(
                0, np.nan
            )
        )

        league_avg = (
            prior[league_col]
            / prior["prior_games"].replace(
                0, np.nan
            )
        )

       
        ratio = (team_avg / league_avg).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)

        prior[f"{market}_ratio"] = ratio

        prior[f"{market}_multiplier"] = (
            1 + SHRINKAGE * (ratio - 1)
        ).clip(
            1 - MAX_ADJUSTMENT,
            1 + MAX_ADJUSTMENT
        ).fillna(1.0)


   
    return prior[
        [
            "game_date",
            "opponent",
            "team_prior_games",
            "PTS_ratio",
            "REB_ratio",
            "PTS_multiplier",
            "REB_multiplier",
        ]
    ]




def weighted_projection(history):
    return (
        0.50 * history[-5:].mean()
        + 0.30 * history[-10:].mean()
        + 0.20 * history[-20:].mean()
    )


def generate_predictions(games, multipliers):
    games = games.merge(
        multipliers,
        on=["game_date", "opponent"],
        how="left",
        validate="many_to_one",
    )

    # No historical matchup information means
    # a neutral multiplier of 1.0.

    for market in MARKETS:
        col = f"{market}_multiplier"
        games[col] = games[col].fillna(1.0)

    games["team_prior_games"] = (
        games["team_prior_games"].fillna(0)
    )

    results = []

    for player_id, group in games.groupby(
        "player_id", sort=False
    ):
        group = group.sort_values(
            ["game_date", "game_id"]
        ).reset_index(drop=True)

        for market, col in MARKETS.items():
            values = group[col].to_numpy(
                dtype=float
            )

            for i in range(
                MIN_HISTORY, len(group)
            ):
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

                baseline = weighted_projection(
                    history
                )

                sd = np.std(history, ddof=1)

                if (
                    not np.isfinite(sd)
                    or sd <= 0
                ):
                    continue

                row = group.iloc[i]

                multiplier = float(
                    row[f"{market}_multiplier"]
                )

                adjusted = baseline * multiplier

                # Both models use the same synthetic lines.
                # These are not historical FanDuel odds.

                base_line = np.floor(
                    baseline
                ) + 0.5

                for offset in [-4, -2, 0, 2, 4]:
                    line = base_line + offset

                    if line < 0.5:
                        continue

                    for side in ["OVER", "UNDER"]:
                        won = int(
                            actual > line
                            if side == "OVER"
                            else actual < line
                        )

                        for model, projection in [
                            ("BASELINE", baseline),
                            ("MATCHUP", adjusted),
                        ]:
                            p_over = 1 - normal_cdf(
                                (line - projection) / sd
                            )

                            probability = (
                                p_over
                                if side == "OVER"
                                else 1 - p_over
                            )

                            results.append(
                                {
                                    "game_date": row["game_date"],
                                    "player_id": player_id,
                                    "market": market,
                                    "side": side,
                                    "line": line,
                                    "model": model,
                                    "opponent": row["opponent"],
                                    "opponent_prior_games": int(
                                        row["team_prior_games"]
                                    ),
                                    "multiplier": (
                                        multiplier
                                        if model == "MATCHUP"
                                        else 1.0
                                    ),
                                    "projection": projection,
                                    "sd": sd,
                                    "raw_probability": probability,
                                    "actual": actual,
                                    "won": won,
                                    "absolute_error": abs(
                                        actual - projection
                                    ),
                                    "squared_error": (
                                        actual - projection
                                    ) ** 2,
                                    "brier": (
                                        probability - won
                                    ) ** 2,
                                    "dataset": (
                                        "EVALUATION"
                                        if row["game_date"]
                                        >= EVALUATION_START
                                        else "CALIBRATION"
                                    ),
                                }
                            )

    return pd.DataFrame(results)


def audit_dataset_dates(games):
    """Print the actual game-date ranges for each dataset."""

    audit = games[["game_date", "game_id"]].copy()
    audit["game_date"] = pd.to_datetime(
        audit["game_date"], errors="coerce"
    )

    audit["dataset"] = np.where(
        audit["game_date"] >= EVALUATION_START,
        "EVALUATION",
        "CALIBRATION",
    )

    summary = (
        audit.groupby("dataset", as_index=False)
        .agg(
            first_game=("game_date", "min"),
            last_game=("game_date", "max"),
            unique_games=("game_id", "nunique"),
        )
    )

    print("\n=== NBA BACKTEST DATE AUDIT ===")
    print(summary.to_string(index=False))
    print("Evaluation cutoff:", EVALUATION_START.date())
    print("===============================\n")

def main():
    print("Starting NBA matchup comparison backtest...")

    games = load_games()
    audit_dataset_dates(games)
    print(f"Player-game records: {len(games):,}")

    team_games = build_team_games(games)

    print(f"Paired team-game records: {len(team_games):,}")

    if team_games.empty:
        raise RuntimeError(
            "No paired team games found."
        )

    multipliers = build_prior_multipliers(
        team_games
    )

    results = generate_predictions(
        games, multipliers
    )

    if results.empty:
        raise RuntimeError(
            "No matchup comparison predictions generated."
        )

    DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    comparison_path = (
        DATA_DIR / "backtest_matchup_comparison.csv"
    )

    results.to_csv(
        comparison_path, index=False
    )

    # OVER only for the summary, so OVER/UNDER
    # versions of the same outcome aren't duplicated.

    over = results[
        results["side"] == "OVER"
    ].copy()

    summary = (
        over.groupby(
            ["dataset", "market", "model"],
            as_index=False,
        )
        .agg(
            predictions=("won", "size"),
            mean_absolute_error=(
                "absolute_error", "mean"
            ),
            root_mean_squared_error=(
                "squared_error",
                lambda s: np.sqrt(s.mean()),
            ),
            brier_score=("brier", "mean"),
            predicted_over_rate=(
                "raw_probability", "mean"
            ),
            actual_over_rate=("won", "mean"),
        )
    )

    summary_path = (
        DATA_DIR / "backtest_matchup_summary.csv"
    )

    summary.to_csv(
        summary_path, index=False
    )

    print()
    print("NBA MATCHUP BACKTEST RESULTS")
    print("=" * 65)
    print(
        f"Comparison predictions: {len(results):,}"
    )
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Full results: {comparison_path}")
    print(f"Summary: {summary_path}")
    print()
    print(
        "NOTE: Synthetic lines, not historical "
        "FanDuel prices. No ROI inference."
    )


if __name__ == "__main__":
    main()
