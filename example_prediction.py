#!/usr/bin/env python3
"""
Example: How to Use the NBA Prediction Model
Demonstrates predicting the outcome of a specific game matchup
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def predict_game_example():
    """
    Example showing how to predict a specific game outcome
    """
    print("🏀 NBA GAME PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Load the trained model
    model_path = "models/modern_l1_enr_efg_model.pkl"
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        print(f"✅ Model loaded: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
        print("   Run 'python run_system.py train' first")
        return
    
    # Load some recent team stats from our dataset for examples
    data_path = "data/processed/nba_games_2000_2025_enriched_rolling.csv"
    df = pd.read_csv(data_path)
    
    # Get some recent games to use as examples
    recent_games = df.tail(10)
    
    print("\n🎯 EXAMPLE PREDICTIONS FROM RECENT GAMES:")
    print("-" * 60)
    
    for i, game in recent_games.iterrows():
        # Calculate eFG% the same way the model trainer does
        home_efg = (game['home_rolling_FGM'] + 0.5 * game['home_rolling_FG3M']) / game['home_rolling_FGA']
        away_efg = (game['away_rolling_FGM'] + 0.5 * game['away_rolling_FG3M']) / game['away_rolling_FGA']
        
        # Extract the features the model needs
        features = np.array([[
            game['home_rolling_ENR'],
            game['away_rolling_ENR'], 
            home_efg,
            away_efg
        ]])
        
        # Scale features and make prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get team names (if available)
        home_team = game.get('home_team', 'Home Team')
        away_team = game.get('away_team', 'Away Team')
        
        # Format prediction
        winner = "🏠 HOME" if prediction == 1 else "✈️ AWAY"
        confidence = max(probability) * 100
        
        print(f"🏀 {away_team} @ {home_team}")
        print(f"   📊 Stats: Home ENR={game['home_rolling_ENR']:.1f}, Away ENR={game['away_rolling_ENR']:.1f}")
        print(f"   📊 eFG%: Home={home_efg:.1%}, Away={away_efg:.1%}")
        print(f"   🎯 Prediction: {winner} wins ({confidence:.1f}% confidence)")
        
        # Show actual result if available
        if 'home_win' in game and not pd.isna(game['home_win']):
            actual = "🏠 HOME" if game['home_win'] == 1 else "✈️ AWAY"
            correct = "✅" if prediction == game['home_win'] else "❌"
            print(f"   📈 Actual: {actual} won {correct}")
        print()

def predict_custom_matchup():
    """
    Example showing how to predict a custom matchup with your own stats
    """
    print("\n🎮 CUSTOM MATCHUP PREDICTION")
    print("=" * 50)
    
    # Load the trained model
    model_path = "models/modern_l1_enr_efg_model.pkl"
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
        return
    
    # Example: Lakers vs Warriors matchup
    print("🏀 Example: Lakers (Home) vs Warriors (Away)")
    print()
    
    # Example team stats (you would get these from current NBA data)
    home_enr = 2.5      # Lakers recent effective net rating
    away_enr = 4.1      # Warriors recent effective net rating  
    home_efg = 0.545    # Lakers effective field goal %
    away_efg = 0.558    # Warriors effective field goal %
    
    print(f"📊 TEAM STATS:")
    print(f"   🏠 Lakers (Home): ENR={home_enr}, eFG%={home_efg:.1%}")
    print(f"   ✈️ Warriors (Away): ENR={away_enr}, eFG%={away_efg:.1%}")
    print()
    
    # Create feature array
    features = np.array([[home_enr, away_enr, home_efg, away_efg]])
    
    # Scale features and make prediction
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Format results
    home_prob = probabilities[1] * 100  # Probability of home win
    away_prob = probabilities[0] * 100  # Probability of away win
    
    print(f"🎯 PREDICTION RESULTS:")
    print(f"   🏠 Lakers win probability: {home_prob:.1f}%")
    print(f"   ✈️ Warriors win probability: {away_prob:.1f}%")
    print()
    
    if prediction == 1:
        print(f"🏆 PREDICTED WINNER: Lakers (Home) - {home_prob:.1f}% confidence")
    else:
        print(f"🏆 PREDICTED WINNER: Warriors (Away) - {away_prob:.1f}% confidence")
    
    # Show feature importance
    print(f"\n📋 MODEL FEATURES:")
    print(f"   The model considers these 4 key factors:")
    print(f"   1. Home team effective net rating: {home_enr}")
    print(f"   2. Away team effective net rating: {away_enr}")
    print(f"   3. Home team effective field goal %: {home_efg:.1%}")
    print(f"   4. Away team effective field goal %: {away_efg:.1%}")

def show_model_info():
    """
    Show information about the trained model
    """
    print("\n🤖 MODEL INFORMATION")
    print("=" * 50)
    
    model_path = "models/modern_l1_enr_efg_model.pkl"
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        print(f"✅ Model Type: {type(model).__name__}")
        print(f"📊 Features: home_rolling_ENR, away_rolling_ENR, home_eFG%, away_eFG%")
        print(f"🎯 Expected Accuracy: ~63% (based on holdout testing)")
        print(f"📈 High Confidence Games: ~74% accuracy")
        print(f"💾 Model File: {model_path}")
        print(f"📅 Training Date: {model_data.get('training_date', 'Unknown')}")
        
        # Show feature coefficients if available
        if hasattr(model, 'coef_'):
            print(f"\n📋 Feature Importance (Coefficients):")
            features = ['home_rolling_ENR', 'away_rolling_ENR', 'home_eFG%', 'away_eFG%']
            for feature, coef in zip(features, model.coef_[0]):
                print(f"   {feature}: {coef:.4f}")
                
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
        print("   Run 'python run_system.py train' first")

if __name__ == "__main__":
    # Run all examples
    predict_game_example()
    predict_custom_matchup()
    show_model_info()
    
    print("\n" + "=" * 60)
    print("🎉 THAT'S IT! Your model is ready to use!")
    print("📚 To use in your own code:")
    print("   1. Load model: joblib.load('models/modern_l1_enr_efg_model.pkl')")
    print("   2. Prepare features: [home_ENR, away_ENR, home_eFG%, away_eFG%]") 
    print("   3. Predict: model.predict(features)")
    print("   4. Get probabilities: model.predict_proba(features)")
    print("=" * 60) 