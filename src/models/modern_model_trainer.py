#!/usr/bin/env python3
"""
Modern Model Trainer for L1 ENR EFG Model
Trains on updated 2000-2025 dataset while preserving proven architecture
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModernModelTrainer:
    """
    Trains your proven L1 ENR EFG model on modern NBA data (2000-2025)
    """
    
    def __init__(self, data_file=None):
        self.data_file = data_file or "data/processed/nba_games_1985_2025_enriched_rolling.csv"
        self.model = None
        self.scaler = None
        self.feature_names = ['home_rolling_ENR', 'away_rolling_ENR', 'home_eFG%', 'away_eFG%']
        self.performance_history = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the modern dataset"""
        try:
            print(f"📂 Loading modern dataset: {self.data_file}")
            df = pd.read_csv(self.data_file, parse_dates=["GAME_DATE"])
            
            print(f"📊 Raw dataset: {len(df)} games")
            print(f"📅 Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
            
            # Calculate eFG% for home and away teams
            df["home_eFG%"] = (df["home_rolling_FGM"] + 0.5 * df["home_rolling_FG3M"]) / df["home_rolling_FGA"]
            df["away_eFG%"] = (df["away_rolling_FGM"] + 0.5 * df["away_rolling_FG3M"]) / df["away_rolling_FGA"]
            
            # Check for required columns
            required_columns = [
                "home_rolling_ENR", "away_rolling_ENR",
                "home_eFG%", "away_eFG%", "home_win"
            ]

            # Validate columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")

            # Clean data
            df_clean = df[required_columns + ['GAME_DATE', 'SEASON']].copy()
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df_clean) == 0:
                raise ValueError("No valid data remaining after cleaning")

            # Sort by date for time series validation
            df_clean = df_clean.sort_values('GAME_DATE').reset_index(drop=True)
            
            print(f"✅ Clean dataset: {len(df_clean)} games")
            print(f"🏠 Home wins: {sum(df_clean['home_win'])}/{len(df_clean)} ({sum(df_clean['home_win'])/len(df_clean):.1%})")
            
            # Prepare features and target
            X = df_clean[self.feature_names].values.astype(np.float64)
            y = df_clean["home_win"].values
            dates = df_clean["GAME_DATE"].values
            seasons = df_clean["SEASON"].values

            return X, y, dates, seasons, self.feature_names

        except Exception as e:
            print(f"❌ Error in data loading: {e}")
            return None, None, None, None, None
    
    def train_model_with_time_series_validation(self, X, y, dates):
        """
        Train model using time series validation (more realistic for NBA predictions)
        """
        print("🤖 Training L1 ENR EFG model with time series validation...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Set up logistic regression with L1 regularization (your proven architecture)
        lr = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )

        # Hyperparameter grid (same as your original model)
        param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}

        # Grid search with time series validation
        print("🔍 Performing hyperparameter optimization...")
        grid_search = GridSearchCV(
            lr, param_grid, 
            scoring='accuracy', 
            cv=tscv,  # Time series validation
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_scaled, y)

        # Best model
        self.model = grid_search.best_estimator_
        print(f"\n✅ Best regularization strength (C): {grid_search.best_params_['C']}")
        print(f"📊 Best time series CV accuracy: {grid_search.best_score_:.4f}")

        # Feature importance (L1 regularization effects)
        print("\n🔍 Selected Features (L1 Regularization):")
        feature_importance = []
        for name, coeff in zip(self.feature_names, self.model.coef_[0]):
            if abs(coeff) > 1e-6:
                print(f"   {name}: {coeff:.4f}")
                feature_importance.append((name, coeff))
        
        return grid_search.best_score_, feature_importance
    
    def evaluate_on_holdout_period(self, X, y, dates, holdout_months=6):
        """
        Evaluate model on most recent holdout period (realistic evaluation)
        """
        print(f"📈 Evaluating on last {holdout_months} months as holdout set...")
        
        # Find cutoff date for holdout period
        max_date = pd.to_datetime(dates).max()
        cutoff_date = max_date - pd.DateOffset(months=holdout_months)
        
        train_mask = pd.to_datetime(dates) <= cutoff_date
        test_mask = pd.to_datetime(dates) > cutoff_date
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"📊 Training set: {len(X_train)} games (up to {cutoff_date.date()})")
        print(f"📊 Test set: {len(X_test)} games (after {cutoff_date.date()})")
        
        if len(X_test) == 0:
            print("⚠️  No test data available")
            return None
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train final model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_prob)
        
        print(f"\n🎯 HOLDOUT EVALUATION RESULTS:")
        print(f"📊 Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"📉 Log Loss: {logloss:.4f}")
        
        # Detailed breakdown
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
        
        # Performance by confidence
        high_conf_mask = np.abs(y_prob - 0.5) > 0.2  # >70% or <30% predictions
        if np.sum(high_conf_mask) > 0:
            high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
            print(f"🔥 High Confidence Accuracy: {high_conf_acc:.4f} ({np.sum(high_conf_mask)} games)")
        
        return {
            'accuracy': accuracy,
            'log_loss': logloss,
            'test_games': len(X_test),
            'high_conf_games': np.sum(high_conf_mask) if 'high_conf_mask' in locals() else 0,
            'high_conf_acc': high_conf_acc if 'high_conf_acc' in locals() else None
        }
    
    def save_model(self, model_path="models/enhanced_l1_enr_efg_model.pkl"):
        """Save the trained model and scaler"""
        if self.model is None or self.scaler is None:
            print("❌ No trained model to save")
            return False
        
        try:
            # Create models directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model components
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'training_date': datetime.now().isoformat(),
                'performance_history': self.performance_history
            }
            
            joblib.dump(model_data, model_path)
            print(f"💾 Model saved: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_path="models/enhanced_l1_enr_efg_model.pkl"):
        """Load a previously trained model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.performance_history = model_data.get('performance_history', [])
            
            print(f"✅ Model loaded: {model_path}")
            print(f"📅 Training date: {model_data.get('training_date', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def run_full_training_pipeline(self, save_model=True):
        """
        Run the complete training pipeline
        """
        print("🚀 STARTING MODERN MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Step 1: Load and preprocess data
        print("STEP 1: Loading and preprocessing data")
        X, y, dates, seasons, feature_names = self.load_and_preprocess_data()
        
        if X is None:
            print("❌ Data loading failed")
            return None
        
        # Step 2: Train model with time series validation
        print("\nSTEP 2: Training model with time series validation")
        cv_score, feature_importance = self.train_model_with_time_series_validation(X, y, dates)
        
        # Step 3: Evaluate on holdout period
        print("\nSTEP 3: Evaluating on recent holdout period")
        holdout_results = self.evaluate_on_holdout_period(X, y, dates)
        
        if holdout_results is None:
            print("❌ Holdout evaluation failed")
            return None
        
        # Step 4: Save model
        if save_model:
            print("\nSTEP 4: Saving trained model")
            self.save_model()
        
        # Store performance history
        performance_record = {
            'training_date': datetime.now().isoformat(),
            'cv_score': cv_score,
            'holdout_accuracy': holdout_results['accuracy'],
            'holdout_log_loss': holdout_results['log_loss'],
            'total_games': len(X),
            'feature_importance': feature_importance
        }
        self.performance_history.append(performance_record)
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"📊 Final model accuracy: {holdout_results['accuracy']:.1%}")
        print(f"📈 Ready for current NBA predictions!")
        
        return performance_record

def main():
    """Run the modern model training"""
    print("🏀 MODERN L1 ENR EFG MODEL TRAINER")
    print("Training your proven model architecture on 2000-2025 data")
    print("="*60)
    
    # Check if modern data exists
    data_file = "data/processed/nba_games_1985_2025_enriched_rolling.csv"
    
    try:
        # Try to load the data file
        test_df = pd.read_csv(data_file, nrows=5)
        print(f"✅ Found modern dataset: {data_file}")
    except FileNotFoundError:
        print(f"❌ Modern dataset not found: {data_file}")
        print("🔄 Please run modern_data_collector.py first to collect the data")
        return
    
    # Initialize trainer
    trainer = ModernModelTrainer(data_file)
    
    # Run training pipeline
    results = trainer.run_full_training_pipeline()
    
    if results:
        print(f"\n✅ SUCCESS!")
        print(f"📊 Your L1 ENR EFG model is now trained on modern NBA data")
        print(f"🎯 Accuracy: {results['holdout_accuracy']:.1%}")
        print(f"💾 Model saved and ready for predictions!")
    else:
        print(f"\n❌ Training failed. Check errors above.")

if __name__ == "__main__":
    main() 