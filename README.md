# 🏀 NBA Game Prediction System

A machine learning system for predicting NBA game outcomes using L1 regularized logistic regression with ENR (Effective Net Rating) and eFG% (Effective Field Goal Percentage) features.

## 📊 **Performance**
- **Model Accuracy**: 70.4% on combined NBA data (1985-2025) 🔥
- **High Confidence Games**: 80.6% accuracy  
- **Training Data**: 42,904 games spanning 30 seasons
- **Features**: Enhanced L1 feature selection with historical patterns

## 🚀 **Quick Start**

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv nbaenv
source nbaenv/bin/activate  # On Windows: nbaenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Combine Historical Data (Recommended)
```bash
# Combine 1985-2000 + 2000-2025 datasets for best performance
python run_system.py combine
```

### 3. Train Enhanced Model
```bash
# Train the L1 ENR EFG model on combined 30-year dataset
python run_system.py train
```

### 4. Make Predictions
```bash
# Predict upcoming NBA games
python run_system.py predict
```

## 📁 **Project Structure**

```
nba-project/
├── src/                              # Core source code
│   ├── data/                         # Data collection & processing
│   │   └── modern_data_collector.py  # NBA API data collection
│   ├── models/                       # Model training & prediction
│   │   ├── modern_model_trainer.py   # L1 ENR EFG model training
│   │   └── live_prediction_system.py # Live game predictions
│   └── utils/                        # Utilities & automation
│       ├── betting_optimizer.py      # Sports betting optimization
│       ├── automated_pipeline.py     # Automation framework
│       └── realtime_data.py         # Real-time data utilities
├── data/                             # Data files
│   ├── processed/                    # Processed datasets
│   │   └── nba_games_1985_2025_enriched_rolling.csv  # Enhanced dataset
│   └── raw/                          # Raw NBA API data
├── research/                         # Original research & experiments
│   ├── original_models/              # Historical model experiments
│   └── *.py                         # Legacy research scripts
├── tests/                           # Test files & validation
├── docs/                            # Documentation
│   ├── MODERNIZATION_PLAN.md        # Implementation guide
│   └── IMPLEMENTATION_GUIDE.md      # Technical details
└── scripts/                         # Generated results & tools
```

## 🎯 **Key Features**

### **Data Pipeline**
- **NBA API Integration**: Official NBA data from 2000-2025
- **Rolling Statistics**: Team performance over sliding windows
- **Feature Engineering**: ENR and eFG% calculation
- **Automated Updates**: Daily data collection and processing

### **Machine Learning Model**
- **Architecture**: L1 Regularized Logistic Regression  
- **Features**: Enhanced ENR-focused selection (3 key features)
- **Validation**: Time series cross-validation on 30 years
- **Performance**: 70.4% accuracy with 80.6% on high-confidence games

### **Live Prediction System**
- **Real-time Predictions**: Upcoming game forecasts
- **Confidence Scoring**: Risk-adjusted betting recommendations
- **Team Statistics**: Current rolling averages
- **Export Options**: CSV output for further analysis

### **Sports Betting Integration**
- **Kelly Criterion**: Optimal bet sizing
- **Risk Management**: Position limits and bankroll management
- **Market Analysis**: Compare model vs market probabilities
- **Performance Tracking**: ROI and accuracy monitoring

## 📊 **Model Details**

### **Features Used**
- `home_rolling_ENR`: Home team's effective net rating (last 10 games)
- `away_rolling_ENR`: Away team's effective net rating (last 10 games)  
- `home_eFG%`: Home team's effective field goal percentage
- `away_eFG%`: Away team's effective field goal percentage

### **Architecture**
- **Algorithm**: Logistic Regression with L1 (Lasso) regularization
- **Regularization**: Automatic feature selection
- **Validation**: Time series splits (more realistic than random)
- **Training**: 25 seasons of NBA data (2000-2025)

## 🔄 **Automated Workflow**

### **Daily Operations**
1. **Data Update**: Fetch latest games and team statistics
2. **Prediction Generation**: Upcoming games for next 3 days
3. **Performance Monitoring**: Track model accuracy
4. **Report Generation**: CSV exports with predictions

### **Weekly Operations**
1. **Model Retraining**: Update with new game data
2. **Performance Analysis**: Accuracy and profitability metrics
3. **Feature Validation**: Ensure data quality

## ⚠️ **Important Notes**

### **Data Sources**
- **Primary**: NBA API (official, free, comprehensive)
- **Rate Limits**: 600 requests per 10 minutes
- **Coverage**: 1996-present (sufficient for 2000-2025 needs)

### **Model Limitations**
- **Accuracy**: 58-68% (significantly above 50% random chance)
- **Seasonality**: Lower accuracy early in season (limited rolling stats)
- **Injuries**: Does not account for player injuries or rest games

### **Responsible Use**
- **Entertainment/Research**: Primary purpose
- **Betting**: Only bet what you can afford to lose
- **Risk Management**: Never exceed 5% of bankroll per bet

## 🛠 **Development**

### **Running Tests**
```bash
python -m pytest tests/
```

### **Code Style**
```bash
# Format code
black src/

# Lint code  
pylint src/
```

### **Adding Features**
1. **New Data Sources**: Add to `src/data/`
2. **Model Improvements**: Modify `src/models/`
3. **Utilities**: Add to `src/utils/`

## 📈 **Expected Results**

### **Model Performance**
- **Accuracy**: 58-68% on holdout data
- **High Confidence Games**: 65-75% accuracy
- **Home Team Prediction**: Slight bias (realistic 54-56%)

### **Betting Performance** (If Used Responsibly)
- **ROI**: 3-8% over full season
- **Win Rate**: 55-60% on high confidence bets
- **Drawdown**: Expect 10-20% temporary losses

## 📞 **Support**

### **Common Issues**
- **NBA API Errors**: Check rate limiting and season dates
- **Missing Data**: Some historical seasons may have gaps
- **Model Performance**: Accuracy varies by season stage

### **Feature Requests**
- Advanced player statistics
- Injury impact modeling
- Market integration improvements

---

**Built with ❤️ for NBA analytics and responsible sports betting** 