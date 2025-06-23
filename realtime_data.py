#!/usr/bin/env python3
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from nba_api.stats.endpoints import (
    leaguegamefinder, 
    teamgamelog, 
    scoreboard,
    boxscoretraditionalv2
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
import time

class RealTimeNBAData:
    """
    Real-time NBA data fetcher for live predictions
    """
    
    def __init__(self):
        self.current_season = self._get_current_season()
        
    def _get_current_season(self):
        """Determine current NBA season based on date"""
        now = datetime.now()
        if now.month >= 10:  # Season starts in October
            return f"{now.year}-{str(now.year + 1)[-2:]}"
        else:
            return f"{now.year - 1}-{str(now.year)[-2:]}"
    
    def get_today_games(self):
        """Get today's scheduled games"""
        try:
            # Get today's games
            today = datetime.now().strftime('%Y-%m-%d')
            board = scoreboard.Scoreboard(game_date=today)
            games_df = board.get_data_frames()[0]  # GameHeader
            
            return games_df
        except Exception as e:
            print(f"Error fetching today's games: {e}")
            return pd.DataFrame()
    
    def get_live_scores(self):
        """Get live scores for ongoing games"""
        try:
            live_board = live_scoreboard.ScoreBoard()
            live_data = live_board.get_dict()
            
            games = []
            for game in live_data['scoreboard']['games']:
                game_info = {
                    'game_id': game['gameId'],
                    'game_status': game['gameStatus'],
                    'home_team': game['homeTeam']['teamName'],
                    'away_team': game['awayTeam']['teamName'],
                    'home_score': game['homeTeam']['score'],
                    'away_score': game['awayTeam']['score'],
                    'period': game.get('period', 0),
                    'game_clock': game.get('gameClock', ''),
                    'game_time_utc': game['gameTimeUTC']
                }
                games.append(game_info)
            
            return pd.DataFrame(games)
        except Exception as e:
            print(f"Error fetching live scores: {e}")
            return pd.DataFrame()
    
    def get_recent_team_games(self, team_id, num_games=10):
        """Get most recent games for a team up to current date"""
        try:
            # Get team's recent games
            team_log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=self.current_season,
                season_type_all_star='Regular Season'
            )
            
            games_df = team_log.get_data_frames()[0]
            
            # Sort by date and get most recent games
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values('GAME_DATE', ascending=False)
            
            return games_df.head(num_games)
            
        except Exception as e:
            print(f"Error fetching recent games for team {team_id}: {e}")
            return pd.DataFrame()
    
    def get_upcoming_games(self, days_ahead=7):
        """Get upcoming games for prediction"""
        upcoming_games = []
        
        for i in range(1, days_ahead + 1):
            future_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            try:
                board = scoreboard.Scoreboard(game_date=future_date)
                games_df = board.get_data_frames()[0]
                if not games_df.empty:
                    games_df['game_date'] = future_date
                    upcoming_games.append(games_df)
                    
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching games for {future_date}: {e}")
                continue
        
        if upcoming_games:
            return pd.concat(upcoming_games, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_latest_completed_games(self, hours_back=24):
        """Get games completed in the last N hours"""
        try:
            # Get games from yesterday and today
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)
            
            completed_games = []
            
            for single_date in pd.date_range(start_date.date(), end_date.date()):
                date_str = single_date.strftime('%Y-%m-%d')
                try:
                    board = scoreboard.Scoreboard(game_date=date_str)
                    games_df = board.get_data_frames()[0]
                    
                    # Filter for completed games
                    if not games_df.empty:
                        # You might need to check game status here
                        games_df['fetch_date'] = date_str
                        completed_games.append(games_df)
                        
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error fetching completed games for {date_str}: {e}")
                    continue
            
            if completed_games:
                return pd.concat(completed_games, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error in get_latest_completed_games: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test the real-time data fetcher
    fetcher = RealTimeNBAData()
    
    print("Today's Games:")
    today_games = fetcher.get_today_games()
    print(today_games[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_STATUS_TEXT']] if not today_games.empty else "No games today")
    
    print("\nLive Scores:")
    live_scores = fetcher.get_live_scores()
    print(live_scores if not live_scores.empty else "No live games")
    
    print("\nUpcoming Games (next 3 days):")
    upcoming = fetcher.get_upcoming_games(days_ahead=3)
    print(upcoming[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']] if not upcoming.empty else "No upcoming games") 