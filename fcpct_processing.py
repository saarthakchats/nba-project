#!/usr/bin/env python3
import pandas as pd

def compute_rolling_enr(raw_csv, window=10):
    """
    The current game’s statistic is excluded (using shift(1)) so that only past games
contribute to the rolling average.

Parameters:
    raw_csv (str): Path to the raw CSV file (e.g., games_2022_23.csv).
    window (int): Number of previous games to use in the rolling average.

Returns:
    pd.DataFrame: The raw DataFrame with two new columns:
                  'ENR' (set equal to PLUS_MINUS) and
                  'rolling_ENR' (the rolling average ENR).
"""
    # Read the raw game logs
    df = pd.read_csv(raw_csv)

    # Convert the GAME_DATE column to datetime format
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Sort by TEAM_ID and GAME_DATE so that rolling calculations are sequential
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

    # Define ENR as the PLUS_MINUS (a proxy for effective net rating)
    # If you have separate offensive and defensive ratings, ENR can be computed as OFF_RTG - DEF_RTG.
    df['ENR'] = df['FG_PCT']

    # For each team, calculate the rolling average ENR over the last 'window' games.
    # The shift(1) ensures the current game's ENR is not used.
    df['rolling_ENR'] = df.groupby('TEAM_ID')['ENR'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )

    return df

def merge_rolling_with_enriched(rolling_df, enriched_csv):
    """ 
    Two merge operations are performed:
  - First, merge the home team's rolling_ENR (renamed as home_rolling_ENR).
  - Then, merge the away team's rolling_ENR (renamed as away_rolling_ENR).

Parameters:
    rolling_df (pd.DataFrame): The DataFrame with computed rolling_ENR.
    enriched_csv (str): Path to the enriched game-level dataset CSV.

Returns:
    pd.DataFrame: The enriched DataFrame with new columns for home_rolling_ENR and away_rolling_ENR.
"""
    # Load the enriched game-level data (ensure that GAME_DATE is parsed as datetime)
    enriched = pd.read_csv(enriched_csv, parse_dates=['GAME_DATE'])

    # Merge for the home team:
    # In the rolling DataFrame, TEAM_ID identifies the team; rename it to match the enriched dataset's home_TEAM_ID.
    home_rolling = rolling_df[['GAME_ID', 'TEAM_ID', 'rolling_ENR']].rename(
        columns={'TEAM_ID': 'home_TEAM_ID', 'rolling_ENR': 'home_rolling_ENR'}
    )
    merged = pd.merge(enriched, home_rolling, on=['GAME_ID', 'home_TEAM_ID'], how='left')

    # Merge for the away team:
    # Do the same renaming for away side.
    away_rolling = rolling_df[['GAME_ID', 'TEAM_ID', 'rolling_ENR']].rename(
        columns={'TEAM_ID': 'away_TEAM_ID', 'rolling_ENR': 'away_rolling_ENR'}
    )
    merged = pd.merge(merged, away_rolling, on=['GAME_ID', 'away_TEAM_ID'], how='left')

    return merged

if __name__ == '__main__':
    raw_csv = 'data/games_2022_23.csv'
    enriched_csv = 'data/enriched_nba_game_data.csv'
    window = 10  # Number of previous games to use in the rolling average
    output_csv = 'data/games_2022_23_enriched_fgpct_rolling.csv'
    
    # Step 1: Compute rolling ENR values from the raw dataset.
    rolling_df = compute_rolling_enr(raw_csv, window=10)
    print("Computed rolling ENR for each team from raw data.")

    # Step 2: Merge the rolling ENR values into the enriched dataset.
    final_df = merge_rolling_with_enriched(rolling_df, enriched_csv)

    # Step 3: Handle missing values (if a team has fewer than 10 prior games, rolling_ENR may be missing)
    # Here we fill missing values with the overall mean of the respective columns.
    final_df['home_rolling_ENR'] = final_df['home_rolling_ENR'].fillna(final_df['home_rolling_ENR'].mean())
    final_df['away_rolling_ENR'] = final_df['away_rolling_ENR'].fillna(final_df['away_rolling_ENR'].mean())

    # Save the final dataset – this file will include GAME_ID, GAME_DATE, home and away information,
    # the target variable (home_win), and both home_rolling_ENR and away_rolling_ENR.
    final_df.to_csv(output_csv, index=False)
    print(f"Final processed dataset saved to {output_csv}")
