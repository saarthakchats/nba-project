# NBA Game Outcome Prediction

A machine learning project that predicts NBA game outcomes using historical data and team performance metrics.

## Project Overview

This project builds predictive models to forecast NBA game results (home team wins/losses) using:
- Historical NBA game data (1950s-2023)
- Rolling averages of team performance metrics
- Various machine learning algorithms (Logistic Regression, SVM)

## Key Features

### Data Collection
- **NBA API Integration**: Fetches historical game data, box scores, and team statistics
- **Multi-season Support**: Handles data from 1950s through 2022-23 season
- **Comprehensive Stats**: Collects traditional and advanced basketball metrics

### Feature Engineering
- **ENR (Effective Net Rating)**: Primary performance metric combining offensive and defensive efficiency
- **Rolling Averages**: 10-game rolling windows for recent performance trends
- **Home/Away Splits**: Separate processing for home and away team statistics
- **Statistical Features**: FG%, rebounds, turnovers, 3-pointers, free throws, steals, blocks, etc.

### Machine Learning Models
- **Logistic Regression**: Multiple variants with different feature sets and regularization
- **Support Vector Machine**: Linear and RBF kernels with hyperparameter tuning
- **Cross-validation**: Grid search for optimal hyperparameters
- **Time-series Validation**: Chronological splits to prevent data leakage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/saarthakchats/nba-project.git
cd nba-project
```

2. Create a virtual environment:
```bash
python -m venv nbaenv
source nbaenv/bin/activate  # On Windows: nbaenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection
```bash
# Fetch historical data for specific season range
python data.py 2020_21-2022_23

# Fetch advanced statistics
python advanced_stats.py
```

### Data Processing
```bash
# Preprocess raw game data into home/away format
python preprocessing.py data/games_2022_23.csv

# Compute rolling averages and advanced metrics
python processing.py
```

### Model Training
```bash
# Train logistic regression model with ENR features
python models/enr.py

# Train SVM model with all features
python models/svm.py

# Train logistic regression with all features
python models/all_features.py
```

## Project Structure

```
nba-project/
├── data/                   # CSV data files
├── models/                 # ML model implementations
│   ├── enr.py             # ENR-based logistic regression
│   ├── svm.py             # Support Vector Machine
│   ├── all_features.py    # Multi-feature logistic regression
│   └── ...                # Additional model variants
├── data.py                # NBA API data fetching
├── preprocessing.py       # Raw data preprocessing
├── processing.py          # Feature engineering
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Data Sources

- **NBA API**: Official NBA statistics and game data
- **Historical Coverage**: 1950s through current season
- **Data Types**: Game logs, box scores, team statistics, advanced metrics

## Contributing

This project is part of a Senior Thesis at Princeton University. Contributions and suggestions are welcome!

## License

This project is for academic research purposes.