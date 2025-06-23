#!/usr/bin/env python3
"""
Live 2025 Playoff Predictor using your trained L1 ENR EFG model
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the trained model from the previous test
class LivePlayoffPredictor:
    """
    Use your L1 ENR EFG model for live 2025 playoff predictions
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['home_rolling_ENR', 'away_rolling_ENR', 'home_eFG%', 'away_eFG%']
        self.setup_model()
        
    def setup_model(self):
        """Setup the model with the same parameters that achieved 66.3% accuracy"""
        print("🤖 Setting up your proven L1 ENR EFG model...")
        
        # Create model with exact same parameters that achieved 66.3% accuracy
        self.model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,
            max_iter=1000,
            random_state=42
        )
        
        # Create scaler
        self.scaler = StandardScaler()
        
        # Load the proven feature importance weights
        # These are the exact coefficients from your successful test
        self.proven_coefficients = np.array([0.1508, -0.1444, 0.2565, -0.2627])
        self.proven_intercept = 0.0  # Will be set when we have training data
        
        print("✅ Model setup complete with proven 66.3% accuracy configuration")
    
    def quick_train_on_existing_data(self):
        """Quickly train on your existing data to set the model weights"""
        try:
            # Load your proven data
            df = pd.read_csv("data/rolling_averages_1985_2000.csv")
            
            # Calculate eFG%
            df['home_eFG%'] = (df['home_rolling_FGM'] + 0.5 * df['home_rolling_FG3M']) / df['home_rolling_FGA']
            df['away_eFG%'] = (df['away_rolling_FGM'] + 0.5 * df['away_rolling_FG3M']) / df['away_rolling_FGA']
            
            # Prepare data
            required_columns = ['home_rolling_ENR', 'away_rolling_ENR', 'home_eFG%', 'away_eFG%', 'home_win']
            df_clean = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()
            
            X = df_clean[self.feature_names].values.astype(np.float64)
            y = df_clean['home_win'].values
            
            # Scale and train
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            print("✅ Model trained on historical data")
            return True
            
        except Exception as e:
            print(f"❌ Could not train on historical data: {e}")
            return False
    
    def predict_playoff_game(self, home_stats, away_stats, home_team_name="Home", away_team_name="Away"):
        """
        Predict a single playoff game
        
        Args:
            home_stats: dict with 'ENR' and 'eFG%' keys
            away_stats: dict with 'ENR' and 'eFG%' keys
            home_team_name: Name of home team
            away_team_name: Name of away team
        """
        
        if self.model is None:
            print("❌ Model not trained")
            return None
        
        # Create feature vector
        features = np.array([
            home_stats['ENR'],
            away_stats['ENR'], 
            home_stats['eFG%'],
            away_stats['eFG%']
        ]).reshape(1, -1)
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        home_win_prob = self.model.predict_proba(features_scaled)[0, 1]
        predicted_winner = "HOME" if home_win_prob > 0.5 else "AWAY"
        confidence = abs(home_win_prob - 0.5) * 2
        
        return {
            'home_team': home_team_name,
            'away_team': away_team_name,
            'home_win_probability': home_win_prob,
            'away_win_probability': 1 - home_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_multiple_games(self, games_list):
        """Predict multiple playoff games"""
        predictions = []
        
        print(f"🔮 Making predictions for {len(games_list)} playoff games...")
        
        for game in games_list:
            prediction = self.predict_playoff_game(
                game['home_stats'], 
                game['away_stats'],
                game.get('home_team', 'Home'),
                game.get('away_team', 'Away')
            )
            
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def display_predictions(self, predictions):
        """Display predictions in a nice format"""
        print("\n" + "="*80)
        print("🏆 2025 NBA PLAYOFF PREDICTIONS")
        print("🤖 Model: L1 ENR EFG (66.3% Proven Accuracy)")
        print("="*80)
        
        for i, pred in enumerate(predictions, 1):
            print(f"\n🏀 GAME {i}: {pred['home_team']} vs {pred['away_team']}")
            print("-" * 50)
            
            winner = pred['predicted_winner']
            prob = pred['home_win_probability'] if winner == "HOME" else pred['away_win_probability']
            confidence_level = "🔥 HIGH" if pred['confidence'] > 0.4 else "📊 MEDIUM" if pred['confidence'] > 0.2 else "⚖️ LOW"
            
            print(f"🎯 PREDICTION: {winner} WINS ({prob:.1%})")
            print(f"🎚️ CONFIDENCE: {confidence_level} ({pred['confidence']:.1%})")
            
            print(f"\n📊 Team Stats:")
            print(f"   🏠 {pred['home_team']}: ENR {pred['home_stats']['ENR']:+.1f}, eFG% {pred['home_stats']['eFG%']:.3f}")
            print(f"   ✈️  {pred['away_team']}: ENR {pred['away_stats']['ENR']:+.1f}, eFG% {pred['away_stats']['eFG%']:.3f}")
            
            # Show what drives the prediction
            enr_advantage = pred['home_stats']['ENR'] - pred['away_stats']['ENR']
            efg_advantage = pred['home_stats']['eFG%'] - pred['away_stats']['eFG%']
            
            print(f"\n🔍 Key Factors:")
            print(f"   • ENR Advantage: {enr_advantage:+.1f} (favors {'HOME' if enr_advantage > 0 else 'AWAY'})")
            print(f"   • eFG% Advantage: {efg_advantage:+.3f} (favors {'HOME' if efg_advantage > 0 else 'AWAY'})")
    
    def simulate_2025_playoff_matchups(self):
        """Simulate some realistic 2025 playoff matchups"""
        print("🏆 Simulating realistic 2025 NBA playoff matchups...")
        
        # Realistic playoff team stats (based on typical playoff team performance)
        playoff_matchups = [
            {
                'home_team': 'Boston Celtics',
                'away_team': 'Miami Heat', 
                'home_stats': {'ENR': 6.2, 'eFG%': 0.562},  # Strong offensive team
                'away_stats': {'ENR': 2.1, 'eFG%': 0.531}   # Solid playoff team
            },
            {
                'home_team': 'Denver Nuggets',
                'away_team': 'Los Angeles Lakers',
                'home_stats': {'ENR': 5.8, 'eFG%': 0.576},  # Elite offensive team
                'away_stats': {'ENR': 3.2, 'eFG%': 0.549}   # Veteran playoff team
            },
            {
                'home_team': 'Milwaukee Bucks',
                'away_team': 'Philadelphia 76ers',
                'home_stats': {'ENR': 4.9, 'eFG%': 0.558},  # Well-rounded team
                'away_stats': {'ENR': 4.1, 'eFG%': 0.544}   # Close matchup
            },
            {
                'home_team': 'Phoenix Suns',
                'away_team': 'Dallas Mavericks',
                'home_stats': {'ENR': 3.8, 'eFG%': 0.551},  # Good home team
                'away_stats': {'ENR': 4.5, 'eFG%': 0.565}   # Strong road team
            },
            {
                'home_team': 'Golden State Warriors',
                'away_team': 'Sacramento Kings',
                'home_stats': {'ENR': 2.9, 'eFG%': 0.573},  # Elite shooting
                'away_stats': {'ENR': 1.8, 'eFG%': 0.547}   # Young playoff team
            }
        ]
        
        return self.predict_multiple_games(playoff_matchups)

def main():
    """Run live playoff predictions"""
    print("🚀 NBA 2025 PLAYOFF PREDICTOR")
    print("Using your proven L1 ENR EFG model (66.3% accuracy)")
    print("="*60)
    
    # Initialize predictor
    predictor = LivePlayoffPredictor()
    
    # Train on historical data
    if not predictor.quick_train_on_existing_data():
        print("❌ Could not initialize model")
        return
    
    # Simulate playoff predictions
    predictions = predictor.simulate_2025_playoff_matchups()
    
    if predictions:
        # Display predictions
        predictor.display_predictions(predictions)
        
        # Save predictions
        results_df = pd.DataFrame(predictions)
        results_df.to_csv('2025_playoff_predictions.csv', index=False)
        
        print(f"\n💾 Predictions saved to '2025_playoff_predictions.csv'")
        print(f"📊 Generated {len(predictions)} playoff game predictions")
        print(f"\n🎉 Your model is ready for the 2025 NBA Playoffs!")
        print(f"💡 Use these predictions responsibly and always bet within your means!")
        
    else:
        print("❌ Could not generate predictions")

if __name__ == "__main__":
    main() 