#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.svm import SVC  # Changed import
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])

        # Check for required columns
        required_columns = [
            "home_rolling_ENR", "away_rolling_ENR",
            "home_rolling_FG_PCT", "home_rolling_REB", "home_rolling_TOV",
            "home_rolling_FG3M", "home_rolling_FTM", "away_rolling_FG_PCT",
            "away_rolling_REB", "away_rolling_TOV", "away_rolling_FG3M", "away_rolling_FTM",
             "home_rolling_OREB", "away_rolling_OREB", 
             "home_rolling_FTA", "away_rolling_FTA", "home_rolling_PTS", "away_rolling_PTS",
              "home_rolling_STL", "away_rolling_STL","home_rolling_PF", "away_rolling_PF", "home_rolling_BLK", "away_rolling_BLK",
                "home_win"
        ]

        # Validate columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Clean data
        df_clean = df[required_columns].copy()
        df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after cleaning")

        # Prepare features and target
        feature_columns = required_columns[:-1]  # Exclude home_win
        X = df_clean[feature_columns].values.astype(np.float64)
        y = df_clean["home_win"].values

        return X, y, feature_columns

    except Exception as e:
        print(f"Error in data loading: {e}")
        return None, None, None

def print_selected_features(coefs, feature_names):
    """Print features with non-zero coefficients"""
    print("\nSelected Features:")
    for name, coeff in zip(feature_names, coefs):
        if abs(coeff) > 1e-6:  # Account for floating point precision
            print(f"{name}: {coeff:.4f}")

if __name__ == "__main__":
    # Load and preprocess data (unchanged)
    csv_path = "../data/rolling_averages_1985_2000.csv"
    X, y, feature_names = load_and_preprocess_data(csv_path)
    
    # Standardize features (unchanged)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set up SVM classifier
    svm = SVC(
        kernel='linear',  # Start with linear kernel for interpretability
        max_iter=5000,    # Increase if convergence warnings occur
        random_state=42
    )

    # Hyperparameter grid (example for linear kernel)
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        # For non-linear kernels, add:
        # "gamma": ['scale', 'auto', 0.1, 1],
        # "kernel": ['rbf', 'poly']
    }

    # Cross-validation and grid search (unchanged structure)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
    grid_search.fit(X_scaled, y)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-val accuracy: {grid_search.best_score_:.4f}")

    # Feature weights (only meaningful for linear kernel)
    if best_model.kernel == 'linear':
        print("\nFeature Weights:")
        for name, weight in zip(feature_names, best_model.coef_[0]):
            print(f"{name}: {weight:.4f}")

    # Final evaluation (unchanged)
    split_idx = int(0.8 * len(X_scaled))
    best_model.fit(X_scaled[:split_idx], y[:split_idx])
    y_pred = best_model.predict(X_scaled[split_idx:])
    
    print("\nTest Performance:")
    print(f"Accuracy: {accuracy_score(y[split_idx:], y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y[split_idx:], y_pred))
