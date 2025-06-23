#!/usr/bin/env python3
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

class SportsBettingOptimizer:
    """
    Advanced sports betting optimizer with Kelly Criterion, 
    bankroll management, and market analysis
    """
    
    def __init__(self, bankroll: float = 1000.0, max_bet_percentage: float = 0.05):
        self.bankroll = bankroll
        self.max_bet_percentage = max_bet_percentage  # Never bet more than 5% of bankroll
        self.bet_history = []
        self.performance_metrics = {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'total_wagered': 0.0,
            'net_profit': 0.0,
            'roi': 0.0
        }
        
        # Odds API configuration (you'll need to sign up for a free key)
        self.odds_api_key = "YOUR_ODDS_API_KEY"  # Replace with actual key
        self.odds_api_base = "https://api.the-odds-api.com/v4"
        
    def get_betting_odds(self, sport='basketball_nba') -> pd.DataFrame:
        """Fetch current betting odds from multiple sportsbooks"""
        try:
            url = f"{self.odds_api_base}/sports/{sport}/odds"
            params = {
                'api_key': self.odds_api_key,
                'regions': 'us',  # US sportsbooks
                'markets': 'h2h',  # Head-to-head (moneyline)
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logging.error(f"Odds API error: {response.status_code}")
                return pd.DataFrame()
            
            odds_data = response.json()
            
            # Parse odds data
            parsed_odds = []
            for game in odds_data:
                game_time = game['commence_time']
                teams = [game['home_team'], game['away_team']]
                
                # Get best odds from multiple bookmakers
                best_home_odds = float('-inf')
                best_away_odds = float('-inf')
                
                for bookmaker in game['bookmakers']:
                    for market in bookmaker['markets']:
                        if market['key'] == 'h2h':
                            outcomes = market['outcomes']
                            for outcome in outcomes:
                                if outcome['name'] == game['home_team']:
                                    best_home_odds = max(best_home_odds, outcome['price'])
                                else:
                                    best_away_odds = max(best_away_odds, outcome['price'])
                
                parsed_odds.append({
                    'game_id': game.get('id', ''),
                    'commence_time': game_time,
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_odds': best_home_odds if best_home_odds != float('-inf') else None,
                    'away_odds': best_away_odds if best_away_odds != float('-inf') else None
                })
            
            return pd.DataFrame(parsed_odds)
            
        except Exception as e:
            logging.error(f"Error fetching betting odds: {e}")
            return pd.DataFrame()
    
    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def decimal_to_implied_probability(self, decimal_odds: float) -> float:
        """Convert decimal odds to implied probability"""
        return 1 / decimal_odds
    
    def calculate_kelly_bet_size(self, model_probability: float, decimal_odds: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion
        Formula: f = (bp - q) / b
        where:
        f = fraction of bankroll to bet
        b = decimal odds - 1 (net odds)
        p = probability of winning
        q = probability of losing (1 - p)
        """
        try:
            b = decimal_odds - 1  # Net odds
            p = model_probability
            q = 1 - p
            
            # Kelly fraction
            kelly_fraction = (b * p - q) / b
            
            # Only bet if Kelly is positive (positive expected value)
            if kelly_fraction <= 0:
                return 0.0
                
            # Apply conservative scaling (fractional Kelly)
            conservative_kelly = kelly_fraction * 0.25  # Use 25% of full Kelly
            
            # Cap at maximum bet percentage
            final_fraction = min(conservative_kelly, self.max_bet_percentage)
            
            return final_fraction
            
        except Exception as e:
            logging.error(f"Error calculating Kelly bet size: {e}")
            return 0.0
    
    def find_betting_opportunities(self, predictions: List[Dict], min_edge: float = 0.05) -> List[Dict]:
        """
        Find profitable betting opportunities by comparing model predictions with market odds
        """
        try:
            opportunities = []
            
            # Get current odds
            market_odds = self.get_betting_odds()
            
            if market_odds.empty:
                logging.warning("No market odds available")
                return opportunities
            
            for prediction in predictions:
                # Find matching game in odds
                home_team_match = market_odds[
                    market_odds['home_team'].str.contains(
                        prediction.get('home_team_name', ''), case=False, na=False
                    ) | market_odds['away_team'].str.contains(
                        prediction.get('away_team_name', ''), case=False, na=False
                    )
                ]
                
                if home_team_match.empty:
                    continue
                
                game_odds = home_team_match.iloc[0]
                
                # Get model prediction
                home_win_prob = prediction['home_win_probability']
                away_win_prob = 1 - home_win_prob
                
                # Convert American odds to decimal
                home_decimal_odds = self.american_to_decimal(game_odds['home_odds']) if game_odds['home_odds'] else None
                away_decimal_odds = self.american_to_decimal(game_odds['away_odds']) if game_odds['away_odds'] else None
                
                if not home_decimal_odds or not away_decimal_odds:
                    continue
                
                # Calculate implied probabilities from market odds
                home_market_prob = self.decimal_to_implied_probability(home_decimal_odds)
                away_market_prob = self.decimal_to_implied_probability(away_decimal_odds)
                
                # Find edges (model probability > market probability + minimum edge)
                home_edge = home_win_prob - home_market_prob
                away_edge = away_win_prob - away_market_prob
                
                # Check for profitable opportunities
                if home_edge > min_edge:
                    kelly_size = self.calculate_kelly_bet_size(home_win_prob, home_decimal_odds)
                    if kelly_size > 0:
                        opportunities.append({
                            'game_id': prediction['game_id'],
                            'game_date': prediction['game_date'],
                            'bet_type': 'HOME_WIN',
                            'team': game_odds['home_team'],
                            'model_probability': home_win_prob,
                            'market_probability': home_market_prob,
                            'edge': home_edge,
                            'decimal_odds': home_decimal_odds,
                            'american_odds': game_odds['home_odds'],
                            'kelly_fraction': kelly_size,
                            'recommended_bet': kelly_size * self.bankroll,
                            'expected_value': (home_win_prob * (home_decimal_odds - 1) - (1 - home_win_prob)) * kelly_size * self.bankroll,
                            'confidence': prediction.get('confidence', 0)
                        })
                
                if away_edge > min_edge:
                    kelly_size = self.calculate_kelly_bet_size(away_win_prob, away_decimal_odds)
                    if kelly_size > 0:
                        opportunities.append({
                            'game_id': prediction['game_id'],
                            'game_date': prediction['game_date'],
                            'bet_type': 'AWAY_WIN',
                            'team': game_odds['away_team'],
                            'model_probability': away_win_prob,
                            'market_probability': away_market_prob,
                            'edge': away_edge,
                            'decimal_odds': away_decimal_odds,
                            'american_odds': game_odds['away_odds'],
                            'kelly_fraction': kelly_size,
                            'recommended_bet': kelly_size * self.bankroll,
                            'expected_value': (away_win_prob * (away_decimal_odds - 1) - (1 - away_win_prob)) * kelly_size * self.bankroll,
                            'confidence': prediction.get('confidence', 0)
                        })
            
            # Sort by expected value
            opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logging.error(f"Error finding betting opportunities: {e}")
            return []
    
    def evaluate_bet_portfolio(self, opportunities: List[Dict], max_simultaneous_bets: int = 5) -> List[Dict]:
        """
        Select optimal portfolio of bets considering correlation and diversification
        """
        try:
            if not opportunities:
                return []
            
            # Filter high-confidence, high-edge opportunities
            filtered_opportunities = [
                opp for opp in opportunities 
                if opp['edge'] > 0.1 and opp['confidence'] > 0.6
            ]
            
            # Limit number of simultaneous bets for risk management
            selected_bets = filtered_opportunities[:max_simultaneous_bets]
            
            # Adjust bet sizes if total allocation exceeds safe limit
            total_allocation = sum(bet['kelly_fraction'] for bet in selected_bets)
            max_total_allocation = 0.20  # Never allocate more than 20% of bankroll total
            
            if total_allocation > max_total_allocation:
                scaling_factor = max_total_allocation / total_allocation
                for bet in selected_bets:
                    bet['kelly_fraction'] *= scaling_factor
                    bet['recommended_bet'] *= scaling_factor
                    bet['expected_value'] *= scaling_factor
            
            return selected_bets
            
        except Exception as e:
            logging.error(f"Error evaluating bet portfolio: {e}")
            return []
    
    def simulate_betting_strategy(self, predictions_history: pd.DataFrame, odds_history: pd.DataFrame) -> Dict:
        """
        Backtest the betting strategy on historical data
        """
        try:
            simulation_results = {
                'initial_bankroll': self.bankroll,
                'final_bankroll': self.bankroll,
                'total_bets': 0,
                'winning_bets': 0,
                'total_wagered': 0.0,
                'total_winnings': 0.0,
                'net_profit': 0.0,
                'roi': 0.0,
                'win_rate': 0.0,
                'avg_bet_size': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
            bankroll_history = [self.bankroll]
            current_bankroll = self.bankroll
            
            # Simulate each betting opportunity
            for _, prediction in predictions_history.iterrows():
                # Find corresponding odds (simplified)
                home_win_prob = prediction['home_win_probability']
                
                # Simulate market odds (in real implementation, use actual historical odds)
                simulated_home_odds = 2.0  # Example decimal odds
                
                # Calculate Kelly bet
                kelly_fraction = self.calculate_kelly_bet_size(home_win_prob, simulated_home_odds)
                
                if kelly_fraction > 0:
                    bet_amount = kelly_fraction * current_bankroll
                    
                    # Simulate bet outcome (using actual result if available)
                    bet_won = prediction.get('actual_home_win', np.random.random() < home_win_prob)
                    
                    simulation_results['total_bets'] += 1
                    simulation_results['total_wagered'] += bet_amount
                    
                    if bet_won:
                        winnings = bet_amount * (simulated_home_odds - 1)
                        current_bankroll += winnings
                        simulation_results['winning_bets'] += 1
                        simulation_results['total_winnings'] += winnings
                    else:
                        current_bankroll -= bet_amount
                    
                    bankroll_history.append(current_bankroll)
            
            # Calculate final metrics
            simulation_results['final_bankroll'] = current_bankroll
            simulation_results['net_profit'] = current_bankroll - self.bankroll
            simulation_results['roi'] = (simulation_results['net_profit'] / self.bankroll) * 100
            simulation_results['win_rate'] = (simulation_results['winning_bets'] / simulation_results['total_bets']) * 100 if simulation_results['total_bets'] > 0 else 0
            simulation_results['avg_bet_size'] = simulation_results['total_wagered'] / simulation_results['total_bets'] if simulation_results['total_bets'] > 0 else 0
            
            # Calculate maximum drawdown
            peak_bankroll = self.bankroll
            max_drawdown = 0
            for bankroll in bankroll_history:
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll
                drawdown = (peak_bankroll - bankroll) / peak_bankroll
                max_drawdown = max(max_drawdown, drawdown)
            
            simulation_results['max_drawdown'] = max_drawdown * 100
            
            return simulation_results
            
        except Exception as e:
            logging.error(f"Error in betting strategy simulation: {e}")
            return {}
    
    def get_betting_recommendations(self, predictions: List[Dict]) -> Dict:
        """
        Get complete betting recommendations with risk management
        """
        try:
            # Find opportunities
            opportunities = self.find_betting_opportunities(predictions)
            
            if not opportunities:
                return {
                    'recommendations': [],
                    'summary': {
                        'total_opportunities': 0,
                        'recommended_bets': 0,
                        'total_stake': 0.0,
                        'expected_profit': 0.0
                    }
                }
            
            # Select optimal portfolio
            recommended_bets = self.evaluate_bet_portfolio(opportunities)
            
            # Calculate summary statistics
            total_stake = sum(bet['recommended_bet'] for bet in recommended_bets)
            expected_profit = sum(bet['expected_value'] for bet in recommended_bets)
            
            summary = {
                'total_opportunities': len(opportunities),
                'recommended_bets': len(recommended_bets),
                'total_stake': total_stake,
                'expected_profit': expected_profit,
                'expected_roi': (expected_profit / total_stake * 100) if total_stake > 0 else 0,
                'bankroll_utilization': (total_stake / self.bankroll * 100)
            }
            
            return {
                'recommendations': recommended_bets,
                'all_opportunities': opportunities,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error generating betting recommendations: {e}")
            return {'recommendations': [], 'summary': {}}

if __name__ == "__main__":
    # Example usage
    optimizer = SportsBettingOptimizer(bankroll=1000.0)
    
    # Example predictions (would come from your model)
    sample_predictions = [
        {
            'game_id': 'sample_game_1',
            'game_date': '2024-01-15',
            'home_team_name': 'Lakers',
            'away_team_name': 'Warriors',
            'home_win_probability': 0.65,
            'confidence': 0.75
        }
    ]
    
    recommendations = optimizer.get_betting_recommendations(sample_predictions)
    print("Betting Recommendations:")
    print(f"Total opportunities: {recommendations['summary'].get('total_opportunities', 0)}")
    print(f"Recommended bets: {recommendations['summary'].get('recommended_bets', 0)}")
    print(f"Expected ROI: {recommendations['summary'].get('expected_roi', 0):.2f}%") 