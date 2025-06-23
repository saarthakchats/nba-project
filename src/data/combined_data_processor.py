#!/usr/bin/env python3
"""
Combined Dataset Processor
Combines historical (1985-2000) and modern (2000-2025) NBA data for enhanced model training
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class CombinedNBADataProcessor:
    def __init__(self):
        self.historical_file = "data/archive/enriched_games_1985_86_1999_00.csv"
        self.modern_file = "data/processed/nba_games_2000_2025_processed.csv"
        self.output_file = "data/processed/nba_games_1985_2025_combined.csv"
        self.output_enriched_file = "data/processed/nba_games_1985_2025_enriched_rolling.csv"
        
    def load_and_standardize_datasets(self):
        """
        Load both datasets and standardize their formats
        """
        print("🏀 COMBINING HISTORICAL AND MODERN NBA DATA")
        print("=" * 60)
        
        # Load historical data (1985-2000)
        print("📂 Loading historical data (1985-2000)...")
        try:
            hist_df = pd.read_csv(self.historical_file)
            print(f"✅ Historical data loaded: {len(hist_df)} games")
            print(f"📅 Date range: {hist_df['GAME_DATE'].min()} to {hist_df['GAME_DATE'].max()}")
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return None, None
        
        # Load modern data (2000-2025)  
        print("\n📂 Loading modern data (2000-2025)...")
        try:
            modern_df = pd.read_csv(self.modern_file)
            print(f"✅ Modern data loaded: {len(modern_df)} games")
            print(f"📅 Date range: {modern_df['GAME_DATE'].min()} to {modern_df['GAME_DATE'].max()}")
        except Exception as e:
            print(f"❌ Error loading modern data: {e}")
            return None, None
        
        # Standardize historical data format
        print("\n🔧 Standardizing data formats...")
        hist_df_std = self.standardize_historical_format(hist_df)
        
        return hist_df_std, modern_df
    
    def standardize_historical_format(self, df):
        """
        Standardize historical data to match modern format
        """
        # Column mapping for historical data
        column_mapping = {
            'SEASON_ID': 'SEASON',
            'home_win': 'home_win'  # Keep as is
        }
        
        # Rename columns to match modern format
        df_std = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_std.columns:
                df_std = df_std.rename(columns={old_col: new_col})
        
        # Ensure all required columns exist
        required_cols = [
            'GAME_ID', 'GAME_DATE', 'SEASON', 'home_TEAM_ID', 'home_TEAM_NAME', 
            'home_PTS', 'home_FGM', 'home_FGA', 'home_FG3M', 'home_FG3A', 
            'home_FTM', 'home_FTA', 'home_OREB', 'home_DREB', 'home_REB',
            'home_AST', 'home_STL', 'home_BLK', 'home_TOV', 'home_PF', 'home_PLUS_MINUS',
            'away_TEAM_ID', 'away_TEAM_NAME', 'away_PTS', 'away_FGM', 'away_FGA', 
            'away_FG3M', 'away_FG3A', 'away_FTM', 'away_FTA', 'away_OREB', 
            'away_DREB', 'away_REB', 'away_AST', 'away_STL', 'away_BLK', 
            'away_TOV', 'away_PF', 'away_PLUS_MINUS', 'home_win'
        ]
        
        # Select only the columns we need (that exist in both datasets)
        available_cols = [col for col in required_cols if col in df_std.columns]
        df_std = df_std[available_cols]
        
        # Add missing SEASON column if needed
        if 'SEASON' not in df_std.columns:
            # Extract season from GAME_DATE
            df_std['GAME_DATE'] = pd.to_datetime(df_std['GAME_DATE'])
            df_std['SEASON'] = df_std['GAME_DATE'].dt.year
            # Adjust for NBA season (Oct-Jun)
            df_std.loc[df_std['GAME_DATE'].dt.month >= 10, 'SEASON'] += 1
        
        return df_std
    
    def combine_datasets(self, hist_df, modern_df):
        """
        Combine the historical and modern datasets
        """
        print("\n🔗 Combining datasets...")
        
        # Ensure both datasets have the same columns
        common_cols = list(set(hist_df.columns) & set(modern_df.columns))
        print(f"📊 Common columns: {len(common_cols)}")
        
        hist_df_clean = hist_df[common_cols].copy()
        modern_df_clean = modern_df[common_cols].copy()
        
        # Combine datasets
        combined_df = pd.concat([hist_df_clean, modern_df_clean], ignore_index=True)
        
        # Sort by date
        combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
        combined_df = combined_df.sort_values('GAME_DATE').reset_index(drop=True)
        
        print(f"✅ Combined dataset: {len(combined_df)} games")
        print(f"📅 Full date range: {combined_df['GAME_DATE'].min()} to {combined_df['GAME_DATE'].max()}")
        print(f"🏠 Home win rate: {combined_df['home_win'].mean():.1%}")
        
        return combined_df
    
    def calculate_rolling_averages(self, df, window=10):
        """
        Calculate rolling averages for all teams
        """
        print(f"\n📊 Calculating {window}-game rolling averages...")
        
        # Sort by team and date
        df_sorted = df.sort_values(['home_TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # Stats to calculate rolling averages for
        stats_cols = [
            'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
            'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'
        ]
        
        # Create combined team-game data for rolling averages
        home_games = df_sorted[['GAME_DATE', 'home_TEAM_ID'] + [f'home_{col}' for col in stats_cols]].copy()
        home_games.columns = ['GAME_DATE', 'TEAM_ID'] + stats_cols
        home_games['is_home'] = 1
        
        away_games = df_sorted[['GAME_DATE', 'away_TEAM_ID'] + [f'away_{col}' for col in stats_cols]].copy()
        away_games.columns = ['GAME_DATE', 'TEAM_ID'] + stats_cols
        away_games['is_home'] = 0
        
        # Combine all games
        all_games = pd.concat([home_games, away_games]).sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # Calculate rolling averages
        rolling_stats = {}
        for team_id in all_games['TEAM_ID'].unique():
            team_games = all_games[all_games['TEAM_ID'] == team_id].copy()
            
            for stat in stats_cols:
                team_games[f'rolling_{stat}'] = team_games[stat].rolling(window=window, min_periods=1).mean()
            
            rolling_stats[team_id] = team_games
        
        # Merge rolling stats back to original dataframe
        print("🔗 Merging rolling stats back to games...")
        
        # Create lookup dictionaries for rolling stats
        home_rolling = {}
        away_rolling = {}
        
        for team_id, team_data in rolling_stats.items():
            home_team_data = team_data[team_data['is_home'] == 1]
            away_team_data = team_data[team_data['is_home'] == 0]
            
            for _, row in home_team_data.iterrows():
                key = (team_id, row['GAME_DATE'])
                home_rolling[key] = {f'home_rolling_{stat}': row[f'rolling_{stat}'] for stat in stats_cols}
            
            for _, row in away_team_data.iterrows():
                key = (team_id, row['GAME_DATE'])
                away_rolling[key] = {f'away_rolling_{stat}': row[f'rolling_{stat}'] for stat in stats_cols}
        
        # Add rolling stats to main dataframe
        for stat in stats_cols:
            df_sorted[f'home_rolling_{stat}'] = np.nan
            df_sorted[f'away_rolling_{stat}'] = np.nan
        
        for idx, row in df_sorted.iterrows():
            home_key = (row['home_TEAM_ID'], row['GAME_DATE'])
            away_key = (row['away_TEAM_ID'], row['GAME_DATE'])
            
            if home_key in home_rolling:
                for stat in stats_cols:
                    df_sorted.at[idx, f'home_rolling_{stat}'] = home_rolling[home_key][f'home_rolling_{stat}']
            
            if away_key in away_rolling:
                for stat in stats_cols:
                    df_sorted.at[idx, f'away_rolling_{stat}'] = away_rolling[away_key][f'away_rolling_{stat}']
        
        # Calculate Effective Net Rating (ENR)
        print("📈 Calculating Effective Net Rating (ENR)...")
        df_sorted['home_rolling_ENR'] = df_sorted['home_rolling_PLUS_MINUS']
        df_sorted['away_rolling_ENR'] = df_sorted['away_rolling_PLUS_MINUS']
        
        return df_sorted
    
    def process_combined_data(self):
        """
        Full pipeline to process and combine datasets
        """
        # Load and standardize datasets
        hist_df, modern_df = self.load_and_standardize_datasets()
        if hist_df is None or modern_df is None:
            return False
        
        # Combine datasets
        combined_df = self.combine_datasets(hist_df, modern_df)
        
        # Save basic combined dataset
        print(f"\n💾 Saving combined dataset...")
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        combined_df.to_csv(self.output_file, index=False)
        print(f"✅ Combined dataset saved: {self.output_file}")
        
        # Calculate rolling averages
        enriched_df = self.calculate_rolling_averages(combined_df)
        
        # Remove rows with NaN values in key columns
        print("\n🧹 Cleaning data...")
        key_cols = ['home_rolling_ENR', 'away_rolling_ENR', 'home_rolling_FGM', 'away_rolling_FGM']
        clean_df = enriched_df.dropna(subset=key_cols)
        
        print(f"📊 Final dataset: {len(clean_df)} games (removed {len(enriched_df) - len(clean_df)} incomplete games)")
        
        # Save enriched dataset
        clean_df.to_csv(self.output_enriched_file, index=False)
        print(f"✅ Enriched dataset saved: {self.output_enriched_file}")
        
        # Print final statistics
        print(f"\n📈 FINAL COMBINED DATASET STATISTICS:")
        print(f"📊 Total games: {len(clean_df):,}")
        print(f"📅 Date range: {clean_df['GAME_DATE'].min().date()} to {clean_df['GAME_DATE'].max().date()}")
        print(f"🏠 Home win rate: {clean_df['home_win'].mean():.1%}")
        print(f"📁 Years covered: {clean_df['GAME_DATE'].dt.year.nunique()}")
        print(f"📋 Columns: {len(clean_df.columns)}")
        
        return True

def main():
    """Run the combined data processing"""
    processor = CombinedNBADataProcessor()
    success = processor.process_combined_data()
    
    if success:
        print(f"\n🎉 SUCCESS! Combined dataset ready for training")
        print(f"🚀 Next step: Update your model trainer to use the combined dataset")
    else:
        print(f"\n❌ Data processing failed")

if __name__ == "__main__":
    main() 