#!/usr/bin/env python3
"""
Simple NBA Game Prediction - Copy & Use This Code
"""

import joblib
import numpy as np

def predict_nba_game(home_enr, away_enr, home_efg_pct, away_efg_pct):
    """
    Predict the outcome of an NBA game
    
    Args:
        home_enr: Home team effective net rating (rolling average)
        away_enr: Away team effective net rating (rolling average) 
        home_efg_pct: Home team effective field goal % (e.g., 0.545 for 54.5%)
        away_efg_pct: Away team effective field goal % (e.g., 0.558 for 55.8%)
    
    Returns:
        dict: Prediction results
    """
    
    # Load the trained model
    model_data = joblib.load('models/enhanced_l1_enr_efg_model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Prepare features
    features = np.array([[home_enr, away_enr, home_efg_pct, away_efg_pct]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Format results
    home_win_prob = probabilities[1] * 100
    away_win_prob = probabilities[0] * 100
    predicted_winner = "HOME" if prediction == 1 else "AWAY"
    confidence = max(probabilities) * 100
    
    return {
        'predicted_winner': predicted_winner,
        'home_win_probability': home_win_prob,
        'away_win_probability': away_win_prob,
        'confidence': confidence
    }

# Example usage
if __name__ == "__main__":
    
    print("🏀 NBA Game Prediction Example")
    print("="*40)
    
    # Example 1: Lakers vs Warriors
    print("\n🏆 Lakers (Home) vs Warriors (Away)")
    result = predict_nba_game(
        home_enr=2.5,      # Lakers recent net rating
        away_enr=4.1,      # Warriors recent net rating
        home_efg_pct=0.545, # Lakers eFG%
        away_efg_pct=0.558  # Warriors eFG%
    )
    
    print(f"🎯 Predicted Winner: {result['predicted_winner']}")
    print(f"📊 Home Win Probability: {result['home_win_probability']:.1f}%")
    print(f"📊 Away Win Probability: {result['away_win_probability']:.1f}%")
    print(f"🔥 Confidence: {result['confidence']:.1f}%")
    
    # Example 2: Close matchup
    print("\n🏆 Celtics (Home) vs Nuggets (Away)")
    result2 = predict_nba_game(
        home_enr=6.2,      # Strong home team
        away_enr=5.8,      # Strong away team  
        home_efg_pct=0.572, # Good shooting
        away_efg_pct=0.565  # Good shooting
    )
    
    print(f"🎯 Predicted Winner: {result2['predicted_winner']}")
    print(f"📊 Home Win Probability: {result2['home_win_probability']:.1f}%")
    print(f"📊 Away Win Probability: {result2['away_win_probability']:.1f}%")
    print(f"🔥 Confidence: {result2['confidence']:.1f}%")
    
    print("\n" + "="*40)
    print("📝 Copy the predict_nba_game() function to use in your code!") 