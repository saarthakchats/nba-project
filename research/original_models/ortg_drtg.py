#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def compute_features_and_train(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
        
        # Check for required columns
        required_columns = [
            "home_rolling_FG_PCT", "away_rolling_FG_PCT",
            "home_rolling_FGA", "away_rolling_FGA", "home_rolling_TOV",
             "away_rolling_TOV", "home_rolling_OREB", "away_rolling_OREB", 
             "home_rolling_FTA", "away_rolling_FTA", "home_rolling_PTS", "away_rolling_PTS", 
             "home_win",
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        df = df.dropna(subset=required_columns)
        df = df.sort_values("GAME_DATE")
        
        # Compute advanced metrics
        # Calculate possessions
        df['home_possessions'] = df['home_rolling_FGA'] - df['home_rolling_OREB'] + df['home_rolling_TOV'] + 0.44 * df['home_rolling_FTA']
        df['away_possessions'] = df['away_rolling_FGA'] - df['away_rolling_OREB'] + df['away_rolling_TOV'] + 0.44 * df['away_rolling_FTA']

        # Calculate Offensive Ratings (points per 100 possessions)
        df['home_OFF_RTG'] = (df['home_rolling_PTS'] / df['home_possessions']) * 100
        df['away_OFF_RTG'] = (df['away_rolling_PTS'] / df['away_possessions']) * 100

        # Calculate Defensive Ratings (points allowed per 100 possessions)
        # Typically, the home team's defensive rating equals the away team's offensive output,
        # and vice versa.
        df['home_DEF_RTG'] = (df['away_rolling_PTS'] / df['home_possessions']) * 100
        df['away_DEF_RTG'] = (df['home_rolling_PTS'] / df['away_possessions']) * 100

        df['home_net_RTG'] = df['home_OFF_RTG'] - df['home_DEF_RTG']
        df['away_net_RTG'] = df['away_OFF_RTG'] - df['away_DEF_RTG']

        df['home_ball_security'] = df['home_rolling_TOV'] + df["away_rolling_STL"]
        df['away_ball_security'] = df['away_rolling_TOV'] + df["home_rolling_STL"]
        # Prepare features and target
        feature_columns = ["home_net_RTG", "away_net_RTG","home_rolling_FTA", "away_rolling_FTA", "home_ball_security", "away_ball_security","home_rolling_PF", "away_rolling_PF"]
        X = df[feature_columns].values
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
