#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import logging
import schedule
import time
from pathlib import Path

from realtime_data import RealTimeNBAData
from processing import compute_rolling_metrics, merge_rolling_with_enriched
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_pipeline.log'),
        logging.StreamHandler()
    ]
)

class AutomatedNBAPipeline:
    """
    Automated pipeline for real-time NBA predictions
    """
    
    def __init__(self, model_dir="models_live", data_dir="data_live"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.data_fetcher = RealTimeNBAData()
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Performance tracking
        self.predictions_log = []
        self.model_performance = {'accuracy': [], 'log_loss': [], 'brier_score': []}
        
    def fetch_and_update_data(self):
        """Fetch latest completed games and update dataset"""
        try:
            logging.info("Fetching latest completed games...")
            
            # Get games completed in last 24 hours
            recent_games = self.data_fetcher.get_latest_completed_games(hours_back=24)
            
            if recent_games.empty:
                logging.info("No new completed games found")
                return False
                
            # Process new games
            processed_games = self._process_new_games(recent_games)
            
            if processed_games is not None and not processed_games.empty:
                # Append to existing dataset
                self._update_training_data(processed_games)
                logging.info(f"Added {len(processed_games)} new games to training data")
                return True
            else:
                logging.info("No processable games found")
                return False
                
        except Exception as e:
            logging.error(f"Error in fetch_and_update_data: {e}")
            return False
    
    def _process_new_games(self, raw_games):
        """Process raw games into training format"""
        try:
            # Save raw games temporarily
            temp_raw_path = self.data_dir / "temp_raw_games.csv"
            raw_games.to_csv(temp_raw_path, index=False)
            
            # Use existing preprocessing pipeline
            from preprocessing import process_games
            temp_enriched_path = self.data_dir / "temp_enriched_games.csv"
            
            enriched_df = process_games(str(temp_raw_path), str(temp_enriched_path))
            
            # Compute rolling metrics
            rolling_df = compute_rolling_metrics(str(temp_raw_path), window=10)
            
            # Merge with enriched data
            final_df = merge_rolling_with_enriched(rolling_df, str(temp_enriched_path))
            
            # Clean up temp files
            temp_raw_path.unlink(missing_ok=True)
            temp_enriched_path.unlink(missing_ok=True)
            
            return final_df
            
        except Exception as e:
            logging.error(f"Error processing new games: {e}")
            return None
    
    def _update_training_data(self, new_data):
        """Append new data to existing training dataset"""
        try:
            main_data_path = self.data_dir / "current_training_data.csv"
            
            if main_data_path.exists():
                existing_data = pd.read_csv(main_data_path, parse_dates=['GAME_DATE'])
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                
                # Remove duplicates based on GAME_ID
                combined_data = combined_data.drop_duplicates(subset=['GAME_ID'], keep='last')
                
                # Sort by date
                combined_data = combined_data.sort_values('GAME_DATE')
                
            else:
                combined_data = new_data
            
            combined_data.to_csv(main_data_path, index=False)
            logging.info(f"Updated training data with {len(combined_data)} total games")
            
        except Exception as e:
            logging.error(f"Error updating training data: {e}")
    
    def retrain_model(self, retrain_threshold=50):
        """Retrain model if enough new data is available"""
        try:
            main_data_path = self.data_dir / "current_training_data.csv"
            
            if not main_data_path.exists():
                logging.warning("No training data found")
                return False
                
            df = pd.read_csv(main_data_path, parse_dates=['GAME_DATE'])
            
            # Check if we have enough new data since last retrain
            last_retrain_path = self.model_dir / "last_retrain_date.txt"
            
            if last_retrain_path.exists():
                with open(last_retrain_path, 'r') as f:
                    last_retrain_date = pd.to_datetime(f.read().strip())
                
                new_data_count = len(df[df['GAME_DATE'] > last_retrain_date])
                
                if new_data_count < retrain_threshold:
                    logging.info(f"Only {new_data_count} new games since last retrain. Threshold: {retrain_threshold}")
                    return False
            
            logging.info("Starting model retraining...")
            
            # Prepare features
            feature_cols = [
                'home_rolling_ENR', 'away_rolling_ENR',
                'home_rolling_FG_PCT', 'away_rolling_FG_PCT',
                'home_rolling_REB', 'away_rolling_REB',
                'home_rolling_TOV', 'away_rolling_TOV',
                'home_rolling_PTS', 'away_rolling_PTS'
            ]
            
            # Filter available columns
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < 2:
                logging.error("Insufficient feature columns available")
                return False
            
            self.feature_columns = available_cols
            
            # Prepare data
            df_clean = df.dropna(subset=available_cols + ['home_win'])
            X = df_clean[available_cols].values
            y = df_clean['home_win'].values
            
            # Time-based split (use last 20% for validation)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model = LogisticRegression(
                penalty='l2', 
                C=1.0, 
                solver='lbfgs', 
                max_iter=1000,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Validate model
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            y_pred = self.model.predict(X_val_scaled)
            
            accuracy = accuracy_score(y_val, y_pred)
            logloss = log_loss(y_val, y_pred_proba)
            brier = brier_score_loss(y_val, y_pred_proba)
            
            logging.info(f"Model retrained - Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}, Brier Score: {brier:.4f}")
            
            # Save model
            self._save_model()
            
            # Update last retrain date
            with open(last_retrain_path, 'w') as f:
                f.write(datetime.now().isoformat())
            
            # Track performance
            self.model_performance['accuracy'].append(accuracy)
            self.model_performance['log_loss'].append(logloss)
            self.model_performance['brier_score'].append(brier)
            
            return True
            
        except Exception as e:
            logging.error(f"Error in model retraining: {e}")
            return False
    
    def _save_model(self):
        """Save the trained model and scaler"""
        try:
            model_path = self.model_dir / "current_model.pkl"
            scaler_path = self.model_dir / "current_scaler.pkl"
            features_path = self.model_dir / "feature_columns.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
                
            logging.info("Model saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load the saved model"""
        try:
            model_path = self.model_dir / "current_model.pkl"
            scaler_path = self.model_dir / "current_scaler.pkl"
            features_path = self.model_dir / "feature_columns.pkl"
            
            if not all([p.exists() for p in [model_path, scaler_path, features_path]]):
                logging.warning("Model files not found")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            logging.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def predict_upcoming_games(self):
        """Make predictions for upcoming games"""
        try:
            if self.model is None:
                if not self.load_model():
                    logging.error("No model available for predictions")
                    return None
            
            # Get upcoming games
            upcoming_games = self.data_fetcher.get_upcoming_games(days_ahead=3)
            
            if upcoming_games.empty:
                logging.info("No upcoming games found")
                return None
            
            predictions = []
            
            for _, game in upcoming_games.iterrows():
                home_team_id = game['HOME_TEAM_ID']
                away_team_id = game['VISITOR_TEAM_ID']
                
                # Get recent performance for both teams
                home_recent = self.data_fetcher.get_recent_team_games(home_team_id, 10)
                away_recent = self.data_fetcher.get_recent_team_games(away_team_id, 10)
                
                if home_recent.empty or away_recent.empty:
                    continue
                
                # Compute features (simplified version)
                home_features = self._compute_team_features(home_recent)
                away_features = self._compute_team_features(away_recent)
                
                if home_features is None or away_features is None:
                    continue
                
                # Create feature vector
                feature_vector = np.array([
                    home_features.get('ENR', 0),
                    away_features.get('ENR', 0),
                    home_features.get('FG_PCT', 0.45),
                    away_features.get('FG_PCT', 0.45),
                    home_features.get('REB', 45),
                    away_features.get('REB', 45),
                    home_features.get('TOV', 15),
                    away_features.get('TOV', 15),
                    home_features.get('PTS', 110),
                    away_features.get('PTS', 110)
                ]).reshape(1, -1)
                
                # Make prediction
                feature_vector_scaled = self.scaler.transform(feature_vector)
                home_win_prob = self.model.predict_proba(feature_vector_scaled)[0, 1]
                
                prediction = {
                    'game_id': game['GAME_ID'],
                    'game_date': game.get('game_date', 'Unknown'),
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_win_probability': home_win_prob,
                    'predicted_winner': 'HOME' if home_win_prob > 0.5 else 'AWAY',
                    'confidence': abs(home_win_prob - 0.5) * 2,
                    'timestamp': datetime.now().isoformat()
                }
                
                predictions.append(prediction)
            
            if predictions:
                # Save predictions
                predictions_df = pd.DataFrame(predictions)
                predictions_path = self.data_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                predictions_df.to_csv(predictions_path, index=False)
                
                logging.info(f"Generated {len(predictions)} predictions")
                
            return predictions
            
        except Exception as e:
            logging.error(f"Error in predict_upcoming_games: {e}")
            return None
    
    def _compute_team_features(self, team_games):
        """Compute rolling features for a team"""
        try:
            if team_games.empty:
                return None
            
            # Compute rolling averages
            features = {
                'ENR': team_games['PLUS_MINUS'].mean(),
                'FG_PCT': team_games['FG_PCT'].mean(),
                'REB': team_games['REB'].mean(),
                'TOV': team_games['TOV'].mean(),
                'PTS': team_games['PTS'].mean()
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error computing team features: {e}")
            return None
    
    def run_daily_update(self):
        """Daily update routine"""
        logging.info("Starting daily update routine...")
        
        # Fetch new data
        has_new_data = self.fetch_and_update_data()
        
        # Retrain if needed
        if has_new_data:
            self.retrain_model(retrain_threshold=20)
        
        # Generate predictions
        predictions = self.predict_upcoming_games()
        
        if predictions:
            logging.info(f"Daily update complete. Generated {len(predictions)} predictions")
        else:
            logging.info("Daily update complete. No predictions generated")
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        logging.info("Starting automated NBA prediction pipeline...")
        
        # Schedule daily updates at 6 AM EST (after most games are complete)
        schedule.every().day.at("06:00").do(self.run_daily_update)
        
        # Schedule prediction updates every 4 hours
        schedule.every(4).hours.do(self.predict_upcoming_games)
        
        # Initial run
        self.run_daily_update()
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    pipeline = AutomatedNBAPipeline()
    
    # For development - run single update
    pipeline.run_daily_update()
    
    # For production - start scheduler
    # pipeline.start_scheduler() 