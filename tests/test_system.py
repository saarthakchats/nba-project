"""
Basic tests for NBA Prediction System components
"""

import unittest
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestSystemComponents(unittest.TestCase):
    """Test basic functionality of system components"""
    
    def test_data_file_exists(self):
        """Test that the main data file exists"""
        data_file = project_root / "data" / "processed" / "nba_games_2000_2025_enriched_rolling.csv"
        self.assertTrue(data_file.exists(), "Main dataset file should exist")
    
    def test_data_file_structure(self):
        """Test that the data file has the expected structure"""
        data_file = project_root / "data" / "processed" / "nba_games_2000_2025_enriched_rolling.csv"
        if data_file.exists():
            df = pd.read_csv(data_file, nrows=10)  # Just read first 10 rows for speed
            
            # Check required columns exist
            required_columns = [
                'home_rolling_ENR', 'away_rolling_ENR',
                'home_eFG%', 'away_eFG%', 'home_won'
            ]
            for col in required_columns:
                self.assertIn(col, df.columns, f"Column '{col}' should exist in dataset")
    
    def test_model_imports(self):
        """Test that model modules can be imported"""
        try:
            from src.models.modern_model_trainer import ModernModelTrainer
            from src.models.live_prediction_system import LiveNBAPredictionSystem
            from src.data.modern_data_collector import NBADataCollector
        except ImportError as e:
            self.fail(f"Failed to import system modules: {e}")
    
    def test_model_initialization(self):
        """Test that model classes can be initialized"""
        try:
            from src.models.modern_model_trainer import ModernModelTrainer
            trainer = ModernModelTrainer()
            self.assertIsNotNone(trainer)
        except Exception as e:
            self.fail(f"Failed to initialize ModernModelTrainer: {e}")

if __name__ == '__main__':
    unittest.main() 