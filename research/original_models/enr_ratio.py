#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
        
        # Required columns for computation
        required_columns = ["home_rolling_ENR", "away_rolling_ENR", "home_win"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Compute the ratio feature: relative advantage in ENR
        epsilon = 1e-6  # small constant to avoid division by zero
        df['rolling_ENR_ratio'] = (df['home_rolling_ENR'] - df['away_rolling_ENR'])
        
        # Construct a clean dataframe with this feature and the target
        df_clean = df[["rolling_ENR_ratio", "home_win"]].copy()
        df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        feature_columns = ["rolling_ENR_ratio"]
        X = df_clean[feature_columns].values.astype(np.float64)
        y = df_clean["home_win"].values
        
        return X, y, feature_columns
        
    except Exception as e:
        print(f"Error in data loading: {e}")
        return None, None, None

if __name__ == "__main__":
    csv_path = "../data/rolling_averages_1985_2000.csv"
    X, y, feature_names = load_and_preprocess_data(csv_path)
    
    if X is None or y is None:
        exit(1)
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} feature")
    
    # Standardize the feature values
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
        solver='liblinear',  # Supports L1 regularization
        max_iter=1000,
        random_state=42
    )
    
    # Hyperparameter grid for C (regularization strength)
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    
    # Use TimeSeriesSplit for cross-validation because data has a time-dependent structure
    cv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(lr, param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    best_model = grid_search.best_estimator_
    print(f"\nBest regularization strength (C): {grid_search.best_params_['C']}")
    print(f"Best cross-val accuracy: {grid_search.best_score_:.4f}")
    
    # Since there's one feature, display its coefficient
    print("\nSelected Feature Weight:")
    print(f"{feature_names[0]}: {best_model.coef_[0][0]:.4f}")
    
    # Final evaluation: split the data (e.g., first 80% as train, remaining as test)
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    print("\nFinal Model Evaluation:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
