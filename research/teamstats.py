import numpy as np
import pandas as pd

from nba_api.stats.endpoints import boxscoreadvancedv2
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# List of game IDs you want to fetch data for
game_ids = ["0022200001", "0022200002", "0022200003"]  # Add more game IDs as needed

# Initialize an empty list to store the team data frames
team_data_frames = []

# Loop through each game ID and fetch the team-level data
for game_id in game_ids:
    # Fetch data for the given game ID
    game_data = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
    
    # Extract the team-level data frame (second DataFrame in the response)
    team_data = game_data.get_data_frames()[1]
    
    # Append the team-level DataFrame to the list
    team_data_frames.append(team_data)

# Concatenate all team data frames into a single DataFrame
all_teams_data = pd.concat(team_data_frames, ignore_index=True)

# Print the concatenated team data frame
print(all_teams_data)