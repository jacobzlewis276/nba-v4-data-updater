from pathlib import Path
from math import erf, sqrt
import numpy as np
import pandas as pd

DATA_DIR = Path('data')
SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26', '2026-27']
MARKETS = {'PTS': 'pts', 'REB': 'reb'}
MIN_HISTORY = 10
MAX_HISTORY = 20
SHRINKAGE = 0.20
MAX_ADJUSTMENT = 0.08
EVALUATION_START = pd.Timestamp('2026-07-01')


def normal_cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def load_games():
    frames = []
    for season in SEASONS:
        path = DATA_DIR / f'game_logs_{season}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}')
        df = pd.read_csv(path, low_memory=False)
        df['season'] = season
        frames.append(df)
    games = pd.concat(frames, ignore_index=True)
    games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce').dt.normalize()
    for col in ['pts', 'reb']:
        games[col] = pd.to_numeric(games[col], errors='coerce')
    games = games.dropna(subset=['game_id', 'game_date', 'player_id', 'team', 'opponent'])
    games = games.drop_duplicates(subset=['player_id', 'game_id'])
    games['team'] = games['team'].astype(str).str.strip().str.upper().replace({'SAS': 'SAN'})
    games['opponent'] = games['opponent'].astype(str).str.strip().str.upper().replace({'SAS': 'SAN'})
    return games.sort_values(['game_date', 'game_id', 'player_id']).reset_index(drop=True)


def build_team_games(games):
    # Aggregate player totals once per team-game. Both teams must be present.
    team_games = (games.groupby(['game_id', 'game_date', 'team', 'opponent'], as_index=False)
                  .agg(pts=('pts', 'sum'), reb=('reb', 'sum')))
    paired = team_games.merge(
        team_games[['game_id', 'team', 'pts', 'reb']].rename(
            columns={'team': 'opponent', 'pts': 'opp_pts', 'reb': 'opp_reb'}),
        on=['game_id', 'opponent'], how='inner', validate='one_to_one')
    paired = paired[paired['team'] != paired['opponent']].copy()
    return paired.sort_values(['game_date', 'game_id', 'team']).reset_index(drop=True)


def build_prior_multipliers(team_games):
    # Freeze all defensive information at the START of each calendar date.
    # This prevents other games on the same date from entering a prediction.
    daily = (team_games.groupby(['game_date', 'team'], as_index=False)
             .agg(opp_pts=('opp_pts', 'sum'), opp_reb=('opp_reb', 'sum'),
                  games=('game_id', 'nunique')))
    league_daily = (daily.groupby('game_date', as_index=False)
                    .agg(league_pts=('opp_pts', 'sum'), league_reb=('opp_reb', 'sum'),
                         league_games=('games', 'sum')))
    league_daily = league_daily.sort_values('game_date')
    league_daily['prior_pts'] = league_daily['league_pts'].cumsum().shift(fill_value=0)
    league_daily['prior_reb'] = league_daily['league_reb'].cumsum().shift(fill_value=0)
    league_daily['prior_games'] = league_daily['league_games'].cumsum().shift(fill_value=0)
    dates = league_daily[['game_date', 'prior_pts', 'prior_reb', 'prior_games']]

    # Team prior sums are computed on their own dates and forward-filled
    # onto every date via asof merge (strictly earlier using allow_exact_matches=False).
    teams = daily.sort_values('game_date').copy()
    for col in ['opp_pts', 'opp_reb', 'games']:
        teams[f'cumulative_{col}'] = teams.groupby('team')[col].cumsum()
    all_teams = sorted(team_games['team'].unique())
    calendar = pd.MultiIndex.from_product(
        [sorted(team_games['game_date'].unique()), all_teams],
        names=['game_date', 'team']).to_frame(index=False)
    # Simpler per-team date-aligned shift: reindex cumulative daily stats,
    # forward-fill, then shift one calendar date to exclude the current date.
    team_frames = []
    for team in all_teams:
        idx = dates['game_date']
        subset = teams[teams['team'] == team].set_index('game_date')
        cum = subset[['cumulative_opp_pts', 'cumulative_opp_reb', 'cumulative_games']]
        cum = cum.reindex(idx).ffill().shift(1).fillna(0)
        cum = cum.rename(columns={
            'cumulative_opp_pts': 'team_prior_pts',
            'cumulative_opp_reb': 'team_prior_reb',
            'cumulative_games': 'team_prior_games'})
        cum['game_date'] = idx.to_numpy()
        cum['opponent'] = team
        team_frames.append(cum.reset_index(drop=True))
    prior = pd.concat(team_frames, ignore_index=True).merge(dates, on='game_date', how='left')
    for market, team_col, league_col in [
        ('PTS', 'team_prior_pts', 'prior_pts'),
        ('REB', 'team_prior_reb', 'prior_reb')]:
        team_avg = prior[team_col] / prior['team_prior_games'].replace(0, np.nan)
        league_avg = prior[league_col] / prior['prior_games'].replace(0, np.nan)
