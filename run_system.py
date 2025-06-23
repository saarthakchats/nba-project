#!/usr/bin/env python3
"""
NBA Prediction System - Main Runner
Provides a simple interface to run different components of the system.
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="NBA Prediction System - Run different components"
    )
    parser.add_argument(
        'component',
        choices=['collect', 'train', 'predict', 'combine', 'evaluate'],
        help='Component to run: collect data, train model, make predictions, combine datasets, or run evaluation'
    )
    
    args = parser.parse_args()
    
    if args.component == 'collect':
        print("🏀 Starting data collection...")
        from src.data.modern_data_collector import main as collect_main
        collect_main()
        
    elif args.component == 'train':
        print("🤖 Starting model training...")
        from src.models.modern_model_trainer import main as train_main
        train_main()
        
    elif args.component == 'predict':
        print("🔮 Making predictions...")
        from src.models.live_prediction_system import main as predict_main
        predict_main()
        
    elif args.component == 'combine':
        print("🔗 Combining historical and modern datasets...")
        from src.data.combined_data_processor import main as combine_main
        combine_main()
        
    elif args.component == 'evaluate':
        print("📊 Running professional model evaluation...")
        from evaluation_demo import main as eval_main
        eval_main()

if __name__ == "__main__":
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    main() 