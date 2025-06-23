#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def compute_features_and_train(csv_path):
    try:
        # Load the dataset
        df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
        
        # Compute advanced metrics
        df["home_eFG%"] = (df["home_rolling_FGM"] + 0.5 * df["home_rolling_FG3M"]) / df["home_rolling_FGA"]
        df["away_eFG%"] = (df["away_rolling_FGM"] + 0.5 * df["away_rolling_FG3M"]) / df["away_rolling_FGA"]
        df["home_AST_TOV"] = df["home_rolling_AST"] / df["home_rolling_TOV"]
        df["away_AST_TOV"] = df["away_rolling_AST"] / df["away_rolling_TOV"]
        df["home_AST_MIN_TOV"] = df["home_rolling_AST"] - df["home_rolling_TOV"]
        df["away_AST_MIN_TOV"] = df["away_rolling_AST"] - df["away_rolling_TOV"]
        
        # Handle invalid values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["home_eFG%", "away_eFG%", "home_AST_TOV", "away_AST_TOV", "home_win"], inplace=True)

        # Prepare features and target
        feature_columns = ["home_AST_TOV", "away_AST_TOV"]
        X = df[feature_columns].values.astype(np.float64)
        y = df["home_win"].values

        # Data validation
        if X.size == 0 or y.size == 0:
            raise ValueError("No valid data available after preprocessing.")
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # L1-regularized logistic regression
        model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
        param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
        
        # Cross-validated grid search
        grid_search = GridSearchCV(model, param_grid, scoring="accuracy", cv=KFold(5, shuffle=True, random_state=42))
        grid_search.fit(X_scaled, y)
        best_model = grid_search.best_estimator_

        # Output results
        print(f"\nBest regularization strength (C): {grid_search.best_params_['C']}")
        print(f"Best cross-val accuracy: {grid_search.best_score_:.4f}")
        
        print("\nSelected Features:")
        for name, coeff in zip(feature_columns, best_model.coef_[0]):
            if abs(coeff) > 1e-6:
                print(f"{name}: {coeff:.4f}")

        # Final evaluation
        split_idx = int(0.8 * len(X_scaled))
        best_model.fit(X_scaled[:split_idx], y[:split_idx])
        y_pred = best_model.predict(X_scaled[split_idx:])
        
        print("\nTest Performance:")
        print(f"Accuracy: {accuracy_score(y[split_idx:], y_pred):.4f}")
        print(f"Bias Term (intercept) {best_model.intercept_[0]:.4f}")
        print("Classification Report:")
        print(classification_report(y[split_idx:], y_pred))

        # Optionally, plot the ROC curve to visualize performance
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    input_csv = "../data/rolling_averages_1985_2000.csv"  # Update with your path
    compute_features_and_train(input_csv)
