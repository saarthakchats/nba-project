#!/usr/bin/env python3
"""
Modern NBA Data Collector (2000-2025)
Collects and processes data to keep your L1 ENR EFG model current
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from nba_api.stats.endpoints import leaguegamefinder, teamgamelog
from nba_api.stats.static import teams
import requests
import os
from tqdm import tqdm

class ModernNBADataCollector:
    """
    Comprehensive data collector for updating your model with 2000-2025 data
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.teams = teams.get_teams()
        self.team_mapping = {team['id']: team for team in self.teams}
        
        # Create directories
        os.makedirs(f"{data_dir}/raw", exist_ok=True)
        os.makedirs(f"{data_dir}/processed", exist_ok=True)
        
        print("🏀 Modern NBA Data Collector initialized")
        print(f"📁 Data directory: {data_dir}")
        print(f"🏟️  NBA teams: {len(self.teams)}")
    
    def get_seasons_to_collect(self, start_year=2000, end_year=2025):
        """Generate list of NBA seasons to collect"""
        seasons = []
        for year in range(start_year, end_year):
            season_id = f"{year}-{str(year+1)[2:]}"  # e.g., "2023-24"
            seasons.append(season_id)
        return seasons
    
    def collect_historical_games(self, seasons=None, max_games_per_call=100):
        """
        Collect all NBA games for specified seasons
        Uses NBA API's leaguegamefinder endpoint
        """
        if seasons is None:
            seasons = self.get_seasons_to_collect(2000, 2025)
        
        print(f"📅 Collecting games for {len(seasons)} seasons: {seasons[0]} to {seasons[-1]}")
        
        all_games = []
        
        for season in tqdm(seasons, desc="Collecting seasons"):
            try:
                print(f"\n🔄 Collecting {season} season...")
                
                # Get games for this season
                gamefinder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable='Regular Season'
                )
                
                games_df = gamefinder.get_data_frames()[0]
                
                if not games_df.empty:
                    games_df['SEASON'] = season
                    all_games.append(games_df)
                    print(f"✅ {season}: {len(games_df)} games collected")
                else:
                    print(f"⚠️  {season}: No games found")
                
                # Rate limiting
                time.sleep(0.6)  # NBA API rate limit
                
            except Exception as e:
                print(f"❌ Error collecting {season}: {e}")
                continue
        
        if all_games:
            combined_df = pd.concat(all_games, ignore_index=True)
            
            # Save raw data
            raw_file = f"{self.data_dir}/raw/nba_games_2000_2025_raw.csv"
            combined_df.to_csv(raw_file, index=False)
            print(f"\n💾 Raw data saved: {raw_file}")
            print(f"📊 Total games collected: {len(combined_df)}")
            
            return combined_df
        else:
            print("❌ No games collected")
            return pd.DataFrame()
    
    def process_games_for_model(self, games_df):
        """
        Process raw games data into the format your model expects
        Creates home/away matchups with team stats
        """
        print("🔧 Processing games for model format...")
        
        # Convert GAME_DATE to datetime
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        
        # Sort by date
        games_df = games_df.sort_values('GAME_DATE').reset_index(drop=True)
        
        # Get unique games (each game appears twice in raw data - once per team)
        unique_games = games_df.groupby(['GAME_ID', 'GAME_DATE']).first().reset_index()
        
        processed_games = []
        
        print(f"🎮 Processing {len(unique_games)} unique games...")
        
        for _, game in tqdm(unique_games.iterrows(), total=len(unique_games), desc="Processing games"):
            try:
                # Get both teams' data for this game
                game_teams = games_df[games_df['GAME_ID'] == game['GAME_ID']]
                
                if len(game_teams) != 2:
                    continue
                
                # Determine home/away based on MATCHUP field
                home_team = None
                away_team = None
                
                for _, team_game in game_teams.iterrows():
                    if 'vs.' in team_game['MATCHUP']:
                        home_team = team_game
                    elif '@' in team_game['MATCHUP']:
                        away_team = team_game
                
                if home_team is None or away_team is None:
                    continue
                
                # Create processed game record
                processed_game = {
                    'GAME_ID': game['GAME_ID'],
                    'GAME_DATE': game['GAME_DATE'],
                    'SEASON': game['SEASON'],
                    
                    # Home team data
                    'home_TEAM_ID': home_team['TEAM_ID'],
                    'home_TEAM_NAME': home_team['TEAM_NAME'],
                    'home_PTS': home_team['PTS'],
                    'home_FGM': home_team['FGM'],
                    'home_FGA': home_team['FGA'],
                    'home_FG3M': home_team['FG3M'],
                    'home_FG3A': home_team['FG3A'],
                    'home_FTM': home_team['FTM'],
                    'home_FTA': home_team['FTA'],
                    'home_OREB': home_team['OREB'],
                    'home_DREB': home_team['DREB'],
                    'home_REB': home_team['REB'],
                    'home_AST': home_team['AST'],
                    'home_STL': home_team['STL'],
                    'home_BLK': home_team['BLK'],
                    'home_TOV': home_team['TOV'],
                    'home_PF': home_team['PF'],
                    'home_PLUS_MINUS': home_team['PLUS_MINUS'],
                    
                    # Away team data
                    'away_TEAM_ID': away_team['TEAM_ID'],
                    'away_TEAM_NAME': away_team['TEAM_NAME'],
                    'away_PTS': away_team['PTS'],
                    'away_FGM': away_team['FGM'],
                    'away_FGA': away_team['FGA'],
                    'away_FG3M': away_team['FG3M'],
                    'away_FG3A': away_team['FG3A'],
                    'away_FTM': away_team['FTM'],
                    'away_FTA': away_team['FTA'],
                    'away_OREB': away_team['OREB'],
                    'away_DREB': away_team['DREB'],
                    'away_REB': away_team['REB'],
                    'away_AST': away_team['AST'],
                    'away_STL': away_team['STL'],
                    'away_BLK': away_team['BLK'],
                    'away_TOV': away_team['TOV'],
                    'away_PF': away_team['PF'],
                    'away_PLUS_MINUS': away_team['PLUS_MINUS'],
                    
                    # Game result
                    'home_win': 1 if home_team['PTS'] > away_team['PTS'] else 0
                }
                
                processed_games.append(processed_game)
                
            except Exception as e:
                print(f"Error processing game {game['GAME_ID']}: {e}")
                continue
        
        processed_df = pd.DataFrame(processed_games)
        
        # Save processed data
        processed_file = f"{self.data_dir}/processed/nba_games_2000_2025_processed.csv"
        processed_df.to_csv(processed_file, index=False)
        print(f"💾 Processed data saved: {processed_file}")
        print(f"📊 Processed games: {len(processed_df)}")
        
        return processed_df
    
    def calculate_rolling_averages(self, processed_df, window=10):
        """
        Calculate rolling averages exactly like your original model
        This is the key to maintaining model compatibility
        """
        print(f"📊 Calculating rolling averages (window={window})...")
        
        # Sort by team and date
        df = processed_df.copy()
        df = df.sort_values(['GAME_DATE']).reset_index(drop=True)
        
        # Stats to calculate rolling averages for
        stats_to_roll = [
            'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
            'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'
        ]
        
        # Get all unique teams
        all_teams = set(df['home_TEAM_ID'].unique()) | set(df['away_TEAM_ID'].unique())
        
        team_rolling_stats = {}
        
        print(f"🏟️  Calculating rolling stats for {len(all_teams)} teams...")
        
        for team_id in tqdm(all_teams, desc="Teams"):
            # Get all games for this team (home and away)
            home_games = df[df['home_TEAM_ID'] == team_id].copy()
            away_games = df[df['away_TEAM_ID'] == team_id].copy()
            
            # Combine and sort by date
            home_games['is_home'] = True
            away_games['is_home'] = False
            
            # Rename columns for consistency
            for stat in stats_to_roll:
                home_games[f'team_{stat}'] = home_games[f'home_{stat}']
                away_games[f'team_{stat}'] = away_games[f'away_{stat}']
            
            team_games = pd.concat([
                home_games[['GAME_ID', 'GAME_DATE', 'is_home'] + [f'team_{stat}' for stat in stats_to_roll]],
                away_games[['GAME_ID', 'GAME_DATE', 'is_home'] + [f'team_{stat}' for stat in stats_to_roll]]
            ]).sort_values('GAME_DATE').reset_index(drop=True)
            
            # Calculate rolling averages
            for stat in stats_to_roll:
                team_games[f'rolling_{stat}'] = team_games[f'team_{stat}'].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)  # Shift to avoid data leakage
            
            # Calculate ENR (Effective Net Rating) - your key feature
            team_games['rolling_ENR'] = team_games['rolling_PLUS_MINUS']
            
            team_rolling_stats[team_id] = team_games
        
        # Now merge back with original data
        enriched_games = []
        
        print("🔗 Merging rolling stats with game data...")
        
        for _, game in tqdm(df.iterrows(), total=len(df), desc="Merging"):
            try:
                game_date = game['GAME_DATE']
                home_team_id = game['home_TEAM_ID']
                away_team_id = game['away_TEAM_ID']
                
                # Get rolling stats for home team
                home_team_stats = team_rolling_stats[home_team_id]
                home_rolling = home_team_stats[
                    home_team_stats['GAME_DATE'] == game_date
                ]
                
                # Get rolling stats for away team
                away_team_stats = team_rolling_stats[away_team_id]
                away_rolling = away_team_stats[
                    away_team_stats['GAME_DATE'] == game_date
                ]
                
                if len(home_rolling) == 0 or len(away_rolling) == 0:
                    continue
                
                home_rolling = home_rolling.iloc[0]
                away_rolling = away_rolling.iloc[0]
                
                # Create enriched game record
                enriched_game = game.to_dict()
                
                # Add home team rolling stats
                for stat in stats_to_roll:
                    enriched_game[f'home_rolling_{stat}'] = home_rolling[f'rolling_{stat}']
                
                # Add away team rolling stats
                for stat in stats_to_roll:
                    enriched_game[f'away_rolling_{stat}'] = away_rolling[f'rolling_{stat}']
                
                # Add ENR specifically
                enriched_game['home_rolling_ENR'] = home_rolling['rolling_ENR']
                enriched_game['away_rolling_ENR'] = away_rolling['rolling_ENR']
                
                enriched_games.append(enriched_game)
                
            except Exception as e:
                print(f"Error enriching game {game['GAME_ID']}: {e}")
                continue
        
        enriched_df = pd.DataFrame(enriched_games)
        
        # Remove rows with NaN rolling averages (first games of season)
        enriched_df = enriched_df.dropna(subset=[
            'home_rolling_ENR', 'away_rolling_ENR', 
            'home_rolling_FGM', 'away_rolling_FGM'
        ])
        
        # Save enriched data
        enriched_file = f"{self.data_dir}/processed/nba_games_2000_2025_enriched_rolling.csv"
        enriched_df.to_csv(enriched_file, index=False)
        print(f"💾 Enriched data saved: {enriched_file}")
        print(f"📊 Enriched games: {len(enriched_df)}")
        
        return enriched_df
    
    def get_current_season_updates(self):
        """
        Get the latest games from current season for regular updates
        """
        current_season = "2024-25"  # Update this as needed
        
        print(f"🔄 Getting current season updates: {current_season}")
        
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=current_season,
                season_type_nullable='Regular Season'
            )
            
            current_games = gamefinder.get_data_frames()[0]
            
            if not current_games.empty:
                current_games['SEASON'] = current_season
                print(f"✅ Current season: {len(current_games)} games found")
                return current_games
            else:
                print("⚠️  No current season games found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error getting current season: {e}")
            return pd.DataFrame()
    
    def run_full_data_collection(self):
        """
        Run the complete data collection and processing pipeline
        """
        print("🚀 STARTING FULL DATA COLLECTION PIPELINE")
        print("="*60)
        
        # Step 1: Collect historical games
        print("STEP 1: Collecting historical games (2000-2025)")
        raw_games = self.collect_historical_games()
        
        if raw_games.empty:
            print("❌ No games collected, stopping pipeline")
            return None
        
        # Step 2: Process games
        print("\nSTEP 2: Processing games for model format")
        processed_games = self.process_games_for_model(raw_games)
        
        if processed_games.empty:
            print("❌ No games processed, stopping pipeline")
            return None
        
        # Step 3: Calculate rolling averages
        print("\nSTEP 3: Calculating rolling averages")
        enriched_games = self.calculate_rolling_averages(processed_games)
        
        if enriched_games.empty:
            print("❌ No games enriched, stopping pipeline")
            return None
        
        print(f"\n🎉 DATA COLLECTION COMPLETE!")
        print(f"📊 Final dataset: {len(enriched_games)} games")
        print(f"📅 Date range: {enriched_games['GAME_DATE'].min()} to {enriched_games['GAME_DATE'].max()}")
        print(f"💾 Ready for model training!")
        
        return enriched_games

if __name__ == "__main__":
    # Initialize collector
    collector = ModernNBADataCollector()
    
    print("🏀 NBA Data Modernization Pipeline")
    print("This will collect and process NBA data from 2000-2025")
    print("for your L1 ENR EFG model")
    print()
    
    # Run full pipeline
    final_data = collector.run_full_data_collection()
    
    if final_data is not None:
        print(f"\n✅ SUCCESS: Your model dataset is now updated!")
        print(f"📈 Use the enriched data file to retrain your model")
        print(f"🎯 Ready for modern NBA predictions!")
    else:
        print(f"\n❌ Pipeline failed. Check errors above.") 