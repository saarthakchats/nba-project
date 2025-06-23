#!/usr/bin/env python3
"""
NBA Prediction System - Main Command Line Interface
"""

import sys
import subprocess
from pathlib import Path

def show_help():
    """Display available commands"""
    print("""
NBA Prediction System - Available Commands:

  collect     Collect and process NBA data (2000-2025)
  combine     Combine historical (1985-2000) and modern (2000-2025) datasets
  train       Train the enhanced model on combined dataset
  predict     Make predictions for upcoming games
  evaluate    Run comprehensive model evaluation
  demo        Launch interactive web demo (Streamlit)
  help        Show this help message

Examples:
  python run_system.py collect
  python run_system.py train
  python run_system.py demo
    """)

def run_collect():
    """Run data collection"""
    print("Collecting NBA data...")
    from src.data.modern_data_collector import ModernNBADataCollector
    
    collector = ModernNBADataCollector()
    collector.collect_all_data()
    print("Data collection completed!")

def run_combine():
    """Run data combination"""
    print("Combining historical and modern datasets...")
    from src.data.combined_data_processor import CombinedDataProcessor
    
    processor = CombinedDataProcessor()
    processor.process_combined_data()
    print("Data combination completed!")

def run_train():
    """Run model training"""
    print("Training enhanced model...")
    from src.models.modern_model_trainer import ModernModelTrainer
    
    trainer = ModernModelTrainer()
    trainer.train_enhanced_model()
    print("Model training completed!")

def run_predict():
    """Run predictions"""
    print("🔄 Making predictions...")
    from src.models.live_prediction_system import LivePredictionSystem
    
    predictor = LivePredictionSystem()
    predictor.predict_upcoming_games()
    print("✅ Predictions completed!")

def run_evaluate():
    """Run model evaluation"""
    print("🔄 Running comprehensive evaluation...")
    from evaluation_demo import ProfessionalModelEvaluator
    
    evaluator = ProfessionalModelEvaluator()
    success = evaluator.run_complete_evaluation()
    
    if success:
        print("✅ Evaluation completed successfully!")
    else:
        print("❌ Evaluation failed!")

def run_demo():
    """Launch interactive Streamlit demo"""
    print("🚀 Launching interactive demo...")
    print("📱 Opening in your default browser...")
    print("🔄 Starting Streamlit server...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit. Please ensure it's installed:")
        print("   pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")

def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "collect":
        run_collect()
    elif command == "combine":
        run_combine()
    elif command == "train":
        run_train()
    elif command == "predict":
        run_predict()
    elif command == "evaluate":
        run_evaluate()
    elif command == "demo":
        run_demo()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    main() 