import sys
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

def fetch_historical_games(season):
    """
    Fetch NBA game data for a single season using the nba_api.
    
    Parameters:
        season (str): Season in the NBA API format (e.g., '1950-51').
        
    Returns:
        pd.DataFrame: Game data for the specified season.
    """
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        games = gamefinder.get_data_frames()[0]
        return games
    except Exception as e:
        print(f"Error fetching data for season {season}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def parse_season_range(season_range):
    """
    Parse a season range string (e.g., '1950_51-2022_23') into a list of NBA API-formatted seasons.
    
    Parameters:
        season_range (str): Input range in the format 'XXXX_XX-XXXX_XX'.
        
    Returns:
        list: List of seasons in NBA API format (e.g., ['1950-51', '1951-52', ...]).
    """
    try:
        # Split input into start and end seasons
        start_season, end_season = season_range.split('-')
        
        # Convert to NBA API format (replace underscores with hyphens)
        start_season_nba = start_season.replace('_', '-')
        end_season_nba = end_season.replace('_', '-')
        
        # Extract start and end years (e.g., '1950-51' → 1950)
        start_year = int(start_season_nba.split('-')[0])
        end_year = int(end_season_nba.split('-')[0])
        
        # Generate all seasons between start_year and end_year inclusive
        seasons = []
        for year in range(start_year, end_year + 1):
            season_str = f"{year}-{str(year + 1)[-2:]}"
            seasons.append(season_str)
            
        return seasons
    
    except Exception as e:
        print(f"Invalid season range format: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python data.py <season_range>")
        print("Example: python data.py 1950_51-2022_23")
        sys.exit(1)

    season_range = sys.argv[1]
    seasons = parse_season_range(season_range)
    
    all_games = []
    for season in seasons:
        print(f"Fetching data for season {season}...")
        season_games = fetch_historical_games(season)
        if not season_games.empty:
            all_games.append(season_games)
    
    if not all_games:
        print("No data fetched. Exiting.")
        sys.exit(1)
        
    # Combine all DataFrames
    combined_df = pd.concat(all_games, ignore_index=True)
    
    # Save to CSV with season range in filename
    output_file = f"games_{season_range.replace('-', '_')}.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
