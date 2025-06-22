import numpy as np
import pandas as pd

from nba_api.stats.endpoints import boxscoreadvancedv2
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# List of game IDs you want to fetch data for
game_ids = ["0022200001", "0022200002", "0022200003"]  # Add more game IDs as needed

# Initialize an empty list to store the data frames
game_data_frames = []

# Loop through each game ID and fetch the data
for game_id in game_ids:
    game_data = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id).get_data_frames()[0]
    game_data_frames.append(game_data)

# Concatenate all data frames into a single data frame
all_games_data = pd.concat(game_data_frames, ignore_index=True)

# Print the concatenated data frame
print(all_games_data)