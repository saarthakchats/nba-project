import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, boxscoretraditionalv2

# Define the seasons you're interested in
seasons = ['1965-66', '1966-67']

# Initialize an empty list to store the game data frames
game_data_frames = []

# Loop through each season
for season in seasons:
    # Get the game log for the season
    game_log = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
    
    # Get the game IDs from the game log
    game_ids = game_log['GAME_ID'].tolist()
    
    # Loop through each game ID and get the box score
    for game_id in game_ids:
        game_data = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[0]
        game_data_frames.append(game_data)

# Concatenate all game data frames into a single data frame
all_games_data = pd.concat(game_data_frames, ignore_index=True)

# Print the concatenated data frame
print(all_games_data)