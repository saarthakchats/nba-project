#!/usr/bin/env python3

import pandas as pd
import sys
import os

def process_games(csv_path, output_path):
    # Read the raw CSV file
    df = pd.read_csv(csv_path)

    # Create a column 'is_home'
    # If MATCHUP contains "vs.", that means the team is playing at home.
    # If it contains "@", then the team is away.
    df['is_home'] = df['MATCHUP'].apply(lambda x: True if 'vs.' in x else False)

    # Define key columns that will remain unchanged for merging
    key_cols = ['GAME_ID', 'GAME_DATE', 'SEASON_ID']

    # Split the data into two DataFrames: home_df and away_df
    home_df = df[df['is_home'] == True].copy()
    away_df = df[df['is_home'] == False].copy()

    # Rename non-key columns in each DataFrame so that their names are unique after merging
    home_df = home_df.rename(columns={col: "home_" + col for col in home_df.columns if col not in key_cols})
    away_df = away_df.rename(columns={col: "away_" + col for col in away_df.columns if col not in key_cols})

    # Merge the home and away DataFrames on the key columns
    merged_df = pd.merge(home_df, away_df, on=key_cols, how='inner')

    # Create a target column "home_win". In our dataset, the home team's win/loss flag is
    # in home_WL. We assume a value of "W" means a win (target = 1) and any other value means 0.
    merged_df['home_win'] = merged_df['home_WL'].apply(lambda x: 1 if x.strip().upper() == 'W' else 0)

    # Optionally, drop any columns that are no longer needed
    merged_df.drop(columns=['home_WL', 'away_WL', 'home_is_home', 'away_is_home'], inplace=True, errors='ignore')

    # Save the resulting enriched game-level dataset to a new CSV file
    merged_df.to_csv(output_path, index=False)
    
    return merged_df

def main():
    if len(sys.argv) != 2:
        print("Usage: python preprocessing.py <input_file>")
        print("Example: python preprocessing.py data/games_2010_11.csv")
        sys.exit(1)

    # Get input file name from command-line arguments
    input_file = sys.argv[1]

    # Ensure the input file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)

    # Generate output file name dynamically based on input file name
    base_name = os.path.basename(input_file)  # Extracts file name (e.g., games_2010_11.csv)
    name_without_extension = os.path.splitext(base_name)[0]  # Removes .csv extension (e.g., games_2010_11)
    output_file = f"data/enriched_{name_without_extension}.csv"  # Generates new name (e.g., enriched_games_2010_11.csv)

    print(f"Processing input file: {input_file}")
    
    try:
        processed_df = process_games(input_file, output_file)
        print(f"Enriched game-level data saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
