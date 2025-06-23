from nba_api.stats.endpoints import teamdashboardbygeneralsplits
import pandas as pd

def fetch_team_advanced_stats(team_id, season):
    """
    Fetch advanced team statistics for a given season using nba_api.
    
    Parameters:
        team_id (int): The ID of the team (e.g., 1610612737 for Atlanta Hawks).
        season (str): The NBA season in the format 'XXXX-XX' (e.g., '2022-23').
        
    Returns:
        pd.DataFrame: A DataFrame containing advanced team statistics.
    """
    dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
    )
    
    stats = dashboard.get_data_frames()[0]  # Extract the first DataFrame
    return stats

# Example usage:
team_id = 1610612737  # Atlanta Hawks
season = "2022-23"
stats_df = fetch_team_advanced_stats(team_id, season)
# print(stats_df.head())


# Save the DataFrame to a CSV file
stats_df.to_csv("data/advanced_stats.csv", index=False)
