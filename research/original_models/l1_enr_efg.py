#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
        
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
        df_clean = df[required_columns].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
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
    # Load and preprocess data
    csv_path = "../data/rolling_averages_1985_2000.csv"
    X, y, feature_names = load_and_preprocess_data(csv_path)
    
    if X is None or y is None:
        exit(1)

    # Data validation
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Standardize features
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        print(f"Scaling error: {e}")
        print("Check for NaN/inf values in features")
        exit(1)

    # Set up logistic regression with L1 regularization
    lr = LogisticRegression(
        penalty='l1',
        solver='liblinear',  # Suitable for L1 regularization
        max_iter=1000,
        random_state=42
    )

    # Hyperparameter grid
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}

    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search
    grid_search = GridSearchCV(lr, param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
    grid_search.fit(X_scaled, y)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"\nBest regularization strength (C): {grid_search.best_params_['C']}")
    print(f"Best cross-val accuracy: {grid_search.best_score_:.4f}")

    # Feature selection results
    print_selected_features(best_model.coef_[0], feature_names)

    # Final evaluation
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    print("\nFinal Model Evaluation:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
