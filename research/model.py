import pandas as pd  

# Load raw data (example structure)  
raw_df = pd.read_csv('nba_games_raw.csv')  
raw_df['GAME_DATE'] = pd.to_datetime(raw_df['GAME_DATE'])  
raw_df = raw_df.sort_values(['TEAM_ID', 'GAME_DATE'])  
# Compute ENR for each team-game  
raw_df['ENR'] = raw_df['OFF_RTG'] - raw_df['DEF_RTG']  

# Group by team and compute rolling ENR (window=10 games, min_periods=1)  
raw_df['rolling_ENR'] = raw_df.groupby('TEAM_ID')['ENR'].transform(  
    lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()  
)  
# Create a home team flag  
raw_df['is_home'] = raw_df['MATCHUP'].str.contains('vs.')  

# Split into home and away dataframes  
home_df = raw_df[raw_df['is_home']].copy()  
away_df = raw_df[~raw_df['is_home']].copy()  

# Rename rolling_ENR columns  
home_df = home_df.rename(columns={'rolling_ENR': 'home_rolling_ENR'})  
away_df = away_df.rename(columns={'rolling_ENR': 'away_rolling_ENR'})  

# Merge home and away stats on GAME_ID and DATE  
merged_df = pd.merge(  
    home_df[['GAME_ID', 'GAME_DATE', 'home_rolling_ENR', 'WL']],  
    away_df[['GAME_ID', 'GAME_DATE', 'away_rolling_ENR']],  
    on=['GAME_ID', 'GAME_DATE'],  
    how='inner'  
)  

# Create target variable: 1 if home team won, 0 otherwise  
merged_df['home_win'] = merged_df['WL'].apply(lambda x: 1 if x == 'W' else 0)  
merged_df = merged_df.drop(columns=['WL'])  
# Convert GAME_DATE to datetime  
merged_df['GAME_DATE'] = pd.to_datetime(merged_df['GAME_DATE'])  

# Split chronologically  
train_df = merged_df[merged_df['GAME_DATE'] < '2023-01-01']  
val_df = merged_df[(merged_df['GAME_DATE'] >= '2023-01-01') & (merged_df['GAME_DATE'] < '2024-10-01')]  
test_df = merged_df[merged_df['GAME_DATE'] >= '2024-10-01']  
