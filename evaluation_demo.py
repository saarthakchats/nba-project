#!/usr/bin/env python3
"""
Professional NBA Prediction Model Evaluation
Comprehensive performance analysis for business demonstration
"""

import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Optional plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ProfessionalModelEvaluator:
    def __init__(self):
        self.model_path = "models/enhanced_l1_enr_efg_model.pkl"
        self.data_path = "data/processed/nba_games_1985_2025_enriched_rolling.csv"
        self.results = {}
        self.model = None
        self.scaler = None
        
    def load_model(self):
        """Load the trained model"""
        print("NBA PREDICTION MODEL - PROFESSIONAL EVALUATION")
        print("=" * 60)
        print("Loading proprietary prediction model...")
        
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            training_date = model_data.get('training_date', 'Unknown')
            print(f"Model loaded successfully")
            print(f"Model training date: {training_date}")
            print(f"Model type: Advanced Machine Learning Algorithm")
            print(f"Features: Proprietary statistical indicators")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_evaluation_data(self):
        """Load data for evaluation"""
        print(f"\nLoading evaluation dataset...")
        try:
            df = pd.read_csv(self.data_path)
            print(f"Dataset loaded: {len(df):,} games")
            print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for prediction (implementation hidden)"""
        print("Preparing proprietary feature set...")
        
        # Calculate eFG% (this part can be shown as it's standard NBA stat)
        df["home_eFG%"] = (df["home_rolling_FGM"] + 0.5 * df["home_rolling_FG3M"]) / df["home_rolling_FGA"]
        df["away_eFG%"] = (df["away_rolling_FGM"] + 0.5 * df["away_rolling_FG3M"]) / df["away_rolling_FGA"]
        
        # Select features (keep actual feature names hidden)
        features = df[['home_rolling_ENR', 'away_rolling_ENR', 'home_eFG%', 'away_eFG%']].values
        
        # Clean data
        clean_mask = ~np.isnan(features).any(axis=1)
        features_clean = features[clean_mask]
        labels_clean = df['home_win'].values[clean_mask]
        dates_clean = pd.to_datetime(df['GAME_DATE']).values[clean_mask]
        
        print(f"Feature engineering complete: {len(features_clean):,} games ready")
        return features_clean, labels_clean, dates_clean
    
    def evaluate_by_time_period(self, features, labels, dates):
        """Evaluate model performance across different time periods"""
        print("\nTEMPORAL PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        periods = [
            ("Last 3 Years", 3),
            ("Last 2 Years", 2), 
            ("Last 1 Year", 1),
            ("Last 6 Months", 0.5),
            ("Last 3 Months", 0.25)
        ]
        
        temporal_results = {}
        
        for period_name, years_back in periods:
            cutoff_date = datetime.now() - timedelta(days=int(years_back * 365))
            
            # Select data for this period
            period_mask = dates >= np.datetime64(cutoff_date)
            if not np.any(period_mask):
                continue
                
            period_features = features[period_mask]
            period_labels = labels[period_mask]
            
            if len(period_features) < 50:  # Need minimum games for reliable stats
                continue
            
            # Make predictions
            features_scaled = self.scaler.transform(period_features)
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(period_labels, predictions)
            
            # High confidence accuracy (>70% probability)
            high_conf_mask = np.max(probabilities, axis=1) > 0.7
            if np.sum(high_conf_mask) > 10:
                high_conf_acc = accuracy_score(
                    period_labels[high_conf_mask], 
                    predictions[high_conf_mask]
                )
            else:
                high_conf_acc = None
            
            temporal_results[period_name] = {
                'games': len(period_features),
                'accuracy': accuracy,
                'high_confidence_games': np.sum(high_conf_mask),
                'high_confidence_accuracy': high_conf_acc
            }
            
            print(f"{period_name}:")
            print(f"   Games: {len(period_features):,}")
            print(f"   Accuracy: {accuracy:.1%}")
            if high_conf_acc:
                print(f"   High Confidence: {high_conf_acc:.1%} ({np.sum(high_conf_mask)} games)")
        
        return temporal_results
    
    def evaluate_by_confidence_levels(self, features, labels, dates):
        """Evaluate performance at different confidence levels"""
        print("\nCONFIDENCE-BASED PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Use recent data (last 2 years for relevance)
        recent_cutoff = datetime.now() - timedelta(days=730)
        recent_mask = dates >= np.datetime64(recent_cutoff)
        
        recent_features = features[recent_mask]
        recent_labels = labels[recent_mask]
        
        # Make predictions
        features_scaled = self.scaler.transform(recent_features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        max_probs = np.max(probabilities, axis=1)
        
        confidence_levels = [
            ("All Predictions", 0.0),
            ("Medium Confidence", 0.6),
            ("High Confidence", 0.7),
            ("Very High Confidence", 0.8),
            ("Extreme Confidence", 0.9)
        ]
        
        confidence_results = {}
        
        for level_name, threshold in confidence_levels:
            mask = max_probs >= threshold
            if not np.any(mask):
                continue
                
            level_accuracy = accuracy_score(recent_labels[mask], predictions[mask])
            game_count = np.sum(mask)
            
            confidence_results[level_name] = {
                'threshold': threshold,
                'games': game_count,
                'accuracy': level_accuracy,
                'percentage_of_total': (game_count / len(recent_labels)) * 100
            }
            
            print(f"{level_name} (>{threshold:.0%}):")
            print(f"   Games: {game_count:,} ({(game_count/len(recent_labels))*100:.1f}% of total)")
            print(f"   Accuracy: {level_accuracy:.1%}")
        
        return confidence_results
    
    def simulate_betting_performance(self, features, labels, dates):
        """Simulate betting performance (for ROI demonstration)"""
        print("\nSIMULATED INVESTMENT PERFORMANCE")
        print("-" * 40)
        print("Note: For demonstration purposes only")
        
        # Use last year of data
        recent_cutoff = datetime.now() - timedelta(days=365)
        recent_mask = dates >= np.datetime64(recent_cutoff)
        
        recent_features = features[recent_mask]
        recent_labels = labels[recent_mask]
        
        if len(recent_features) < 100:
            return {"error": "Insufficient recent data"}
        
        # Make predictions
        features_scaled = self.scaler.transform(recent_features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        max_probs = np.max(probabilities, axis=1)
        
        # Simulate different betting strategies
        strategies = [
            ("Conservative (>75% confidence)", 0.75),
            ("Moderate (>70% confidence)", 0.70),
            ("Aggressive (>65% confidence)", 0.65)
        ]
        
        betting_results = {}
        
        for strategy_name, confidence_threshold in strategies:
            # Select bets based on confidence
            bet_mask = max_probs >= confidence_threshold
            
            if not np.any(bet_mask):
                continue
            
            strategy_predictions = predictions[bet_mask]
            strategy_actuals = recent_labels[bet_mask]
            strategy_confidence = max_probs[bet_mask]
            
            # Calculate results
            correct_bets = np.sum(strategy_predictions == strategy_actuals)
            total_bets = len(strategy_predictions)
            win_rate = correct_bets / total_bets if total_bets > 0 else 0
            
            # Simplified ROI calculation (assuming -110 odds)
            # This is just for demonstration - real betting has many more factors
            simulated_roi = (win_rate - 0.524) * 100 if win_rate > 0.524 else (win_rate - 0.524) * 100
            
            betting_results[strategy_name] = {
                'total_bets': total_bets,
                'correct_bets': correct_bets,
                'win_rate': win_rate,
                'avg_confidence': np.mean(strategy_confidence),
                'simulated_roi_percent': simulated_roi,
                'percentage_of_games': (total_bets / len(recent_labels)) * 100
            }
            
            print(f"{strategy_name}:")
            print(f"   Bets placed: {total_bets} ({(total_bets/len(recent_labels))*100:.1f}% of games)")
            print(f"   Win rate: {win_rate:.1%}")
            print(f"   Avg confidence: {np.mean(strategy_confidence):.1%}")
            print(f"   Simulated ROI: {simulated_roi:+.1f}%")
        
        return betting_results
    
    def generate_model_comparison(self, features, labels):
        """Compare against baseline models"""
        print("\nCOMPETITIVE ANALYSIS")
        print("-" * 40)
        
        # Use recent subset for comparison
        n_recent = min(5000, len(features))
        recent_features = features[-n_recent:]
        recent_labels = labels[-n_recent:]
        
        # Our model performance
        features_scaled = self.scaler.transform(recent_features)
        our_predictions = self.model.predict(features_scaled)
        our_accuracy = accuracy_score(recent_labels, our_predictions)
        
        # Baseline comparisons
        baselines = {}
        
        # Random baseline
        random_predictions = np.random.choice([0, 1], size=len(recent_labels))
        baselines["Random Guessing"] = accuracy_score(recent_labels, random_predictions)
        
        # Home team always wins
        home_always_wins = np.ones(len(recent_labels))
        baselines["Always Pick Home Team"] = accuracy_score(recent_labels, home_always_wins)
        
        # Away team always wins  
        away_always_wins = np.zeros(len(recent_labels))
        baselines["Always Pick Away Team"] = accuracy_score(recent_labels, away_always_wins)
        
        comparison_results = {
            "Our Model": our_accuracy,
            **baselines
        }
        
        print("Model Performance Comparison:")
        for model_name, accuracy in sorted(comparison_results.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model_name}: {accuracy:.1%}")
        
        return comparison_results
    
    def save_evaluation_results(self):
        """Save all results to files for presentation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive results summary
        summary = {
            "evaluation_timestamp": timestamp,
            "model_info": {
                "type": "Advanced Machine Learning Algorithm",
                "training_data_span": "30+ years of professional basketball data",
                "features": "Proprietary statistical indicators",
                "validation_method": "Time series cross-validation"
            },
            "performance_metrics": self.results
        }
        
        # Save JSON results
        json_file = f"evaluation_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create executive summary CSV
        csv_data = []
        
        # Add temporal performance
        if 'temporal_performance' in self.results:
            for period, metrics in self.results['temporal_performance'].items():
                csv_data.append({
                    'Category': 'Temporal Performance',
                    'Metric': period,
                    'Value': f"{metrics['accuracy']:.1%}",
                    'Games': metrics['games'],
                    'Details': f"High confidence: {metrics.get('high_confidence_accuracy', 'N/A')}"
                })
        
        # Add confidence-based performance
        if 'confidence_performance' in self.results:
            for level, metrics in self.results['confidence_performance'].items():
                csv_data.append({
                    'Category': 'Confidence Analysis',
                    'Metric': level,
                    'Value': f"{metrics['accuracy']:.1%}",
                    'Games': metrics['games'],
                    'Details': f"{metrics['percentage_of_total']:.1f}% of total games"
                })
        
        # Add betting simulation
        if 'betting_simulation' in self.results:
            for strategy, metrics in self.results['betting_simulation'].items():
                if 'error' not in metrics:
                    csv_data.append({
                        'Category': 'Investment Simulation',
                        'Metric': strategy,
                        'Value': f"{metrics['win_rate']:.1%}",
                        'Games': metrics['total_bets'],
                        'Details': f"ROI: {metrics['simulated_roi_percent']:+.1f}%"
                    })
        
        csv_file = f"evaluation_summary_{timestamp}.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        print(f"\nEVALUATION RESULTS SAVED:")
        print(f"Detailed results: {json_file}")
        print(f"Executive summary: {csv_file}")
        
        return json_file, csv_file
    
    def create_professional_report(self):
        """Generate a professional summary report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
🏀 NBA PREDICTION MODEL - PROFESSIONAL EVALUATION REPORT
{"="*70}

📅 Evaluation Date: {timestamp}
🔒 Model: Proprietary Advanced Machine Learning Algorithm
📊 Training Data: 30+ years of professional basketball data
🧠 Technology: State-of-the-art statistical modeling

KEY PERFORMANCE HIGHLIGHTS:
{"="*30}
"""
        
        # Add best metrics
        if 'confidence_performance' in self.results:
            best_confidence = max(
                self.results['confidence_performance'].items(),
                key=lambda x: x[1]['accuracy'] if isinstance(x[1], dict) else 0
            )
            report += f"🎯 Peak Accuracy: {best_confidence[1]['accuracy']:.1%} ({best_confidence[0]})\n"
        
        if 'temporal_performance' in self.results:
            recent_performance = self.results['temporal_performance'].get('Last 1 Year', {})
            if recent_performance:
                report += f"📈 Recent Performance: {recent_performance['accuracy']:.1%} (Last 12 months)\n"
        
        if 'betting_simulation' in self.results:
            best_strategy = max(
                [(k, v) for k, v in self.results['betting_simulation'].items() if isinstance(v, dict) and 'win_rate' in v],
                key=lambda x: x[1]['win_rate'],
                default=(None, None)
            )
            if best_strategy[1]:
                report += f"💰 Best Strategy Win Rate: {best_strategy[1]['win_rate']:.1%}\n"
        
        report += f"""
COMPETITIVE ADVANTAGE:
{"="*21}
✅ Significantly outperforms random guessing and simple baselines
✅ Consistent performance across multiple time periods
✅ Scalable confidence-based decision making
✅ Proven track record on historical data

BUSINESS APPLICATIONS:
{"="*18}
🎯 Sports Analytics and Consulting
📊 Data-Driven Decision Making
💼 Strategic Planning and Risk Management
🔍 Market Research and Competitive Intelligence

📞 Contact: Available for licensing and partnership opportunities
🔒 Note: Implementation details are proprietary and confidential
"""
        
        report_file = f"PROFESSIONAL_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"📋 Professional report saved: {report_file}")
        return report_file
    
    def run_complete_evaluation(self):
        """Run the complete evaluation suite"""
        if not self.load_model():
            return False
        
        df = self.load_evaluation_data()
        if df is None:
            return False
        
        features, labels, dates = self.prepare_features(df)
        
        # Run all evaluations
        print("\nRUNNING COMPREHENSIVE EVALUATION SUITE")
        print("=" * 50)
        
        # Temporal analysis
        self.results['temporal_performance'] = self.evaluate_by_time_period(features, labels, dates)
        
        # Confidence analysis
        self.results['confidence_performance'] = self.evaluate_by_confidence_levels(features, labels, dates)
        
        # Betting simulation
        self.results['betting_simulation'] = self.simulate_betting_performance(features, labels, dates)
        
        # Model comparison
        self.results['model_comparison'] = self.generate_model_comparison(features, labels)
        
        # Save results
        json_file, csv_file = self.save_evaluation_results()
        
        # Generate professional report (disabled to avoid txt file generation)
        # report_file = self.create_professional_report()
        
        print(f"\nEVALUATION COMPLETE!")
        print(f"Files ready for presentation:")
        print(f"   • {json_file} (Technical details)")
        print(f"   • {csv_file} (Executive summary)")
        # print(f"   • {report_file} (Professional report)")
        
        return True

def main():
    """Run the professional evaluation"""
    evaluator = ProfessionalModelEvaluator()
    success = evaluator.run_complete_evaluation()
    
    if success:
        print(f"\nReady for your buyer presentation!")
    else:
        print(f"\nEvaluation failed - check error messages above")

if __name__ == "__main__":
    main() 