#!/usr/bin/env python3
import pandas as pd

def compute_rolling_metrics(raw_csv, window=10):
    """Calculate rolling averages for key metrics, handling column name variations."""
    df = pd.read_csv(raw_csv)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    
    # Column name mappings (adjust according to your raw data)
    COLUMN_MAP = {
        'FG_PCT': 'FG_PCT',      # Common alternative name
        'FG3M': 'FG3M',        # Three-point made
        'FTM': 'FTM',         # Free throws made
        'REB': 'REB',         # Rebounds
        'TOV': 'TOV',          # Turnovers
        'AST': 'AST',         # Assists
    }
    
    # Calculate ENR (PLUS_MINUS is used directly)
    df['ENR'] = df['PLUS_MINUS']
    
    # Compute rolling averages for each metric
    metrics = ['ENR', 'FG_PCT', 'REB', 'TOV', 'FG3M', 'FTM', 'AST', 'FGA', 'FG3A', 'FTA', 'FGM', 'OREB', 'DREB', 'PF', 'STL', 'BLK', 'PTS']
    for metric in metrics:
        source_col = COLUMN_MAP.get(metric, metric)
        if source_col not in df.columns:
            raise ValueError(f"Column '{source_col}' not found in raw data.")
        
        df[f'rolling_{metric}'] = df.groupby('TEAM_ID')[source_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).median()
        )
    
    return df

def merge_rolling_with_enriched(rolling_df, enriched_csv):
    """Merge rolling averages into enriched dataset."""
    enriched = pd.read_csv(enriched_csv, parse_dates=['GAME_DATE'])
    metrics = ['ENR', 'FG_PCT', 'REB', 'TOV', 'FG3M', 'FTM', 'AST', 'FGA', 'FG3A', 'FTA', 'FGM', 'OREB', 'DREB', 'PF', 'STL', 'BLK', 'PTS']
    
    # Merge home team metrics
    home_rolling = rolling_df[['GAME_ID', 'TEAM_ID'] + [f'rolling_{m}' for m in metrics]]
    home_rolling = home_rolling.rename(
        columns={'TEAM_ID': 'home_TEAM_ID', **{f'rolling_{m}': f'home_rolling_{m}' for m in metrics}}
    )
    merged = pd.merge(enriched, home_rolling, on=['GAME_ID', 'home_TEAM_ID'], how='left')
    
    # Merge away team metrics
    away_rolling = rolling_df[['GAME_ID', 'TEAM_ID'] + [f'rolling_{m}' for m in metrics]]
    away_rolling = away_rolling.rename(
        columns={'TEAM_ID': 'away_TEAM_ID', **{f'rolling_{m}': f'away_rolling_{m}' for m in metrics}}
    )
    merged = pd.merge(merged, away_rolling, on=['GAME_ID', 'away_TEAM_ID'], how='left')
    
    # Fill missing values with column means
    for col in merged.columns:
        if 'rolling_' in col:
            merged[col] = merged[col].fillna(merged[col].mean())
    
    return merged

if __name__ == '__main__':
    raw_csv = 'data/games_1985_86_1999_00.csv'  # Update to your raw data path
    enriched_csv = 'data/enriched_games_1985_86_1999_00.csv'
    output_csv = 'data/rolling_averages_1985_2000.csv'
    window = 10
    rolling_df = compute_rolling_metrics(raw_csv, window=window)
    final_df = merge_rolling_with_enriched(rolling_df, enriched_csv)
    final_df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")
