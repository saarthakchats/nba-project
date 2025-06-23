#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(csv_path):
    # Load the enriched dataset that contains home_rolling_ENR, away_rolling_ENR, and home_win
    df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
    # Drop rows with missing values in key columns
    df = df.dropna(subset=["home_rolling_ENR", "away_rolling_ENR", "home_win"])
    # Sort the data chronologically
    df = df.sort_values("GAME_DATE")
    # Create feature matrix and target vector
    X = df[["home_rolling_ENR", "away_rolling_ENR"]].values
    y = df["home_win"].values
    return X, y

if __name__ == "__main__":
    # Load and preprocess data
    csv_path = "data/games_2022_23_enriched_rolling.csv"
    X, y = load_and_preprocess_data(csv_path)
    # Standardize the features. Cross-validation should be run after scaling the features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set up the logistic regression model.
    # We use L2 regularization (the default). The hyperparameter C determines regularization strength.
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)

    # Set up a grid of possible C values to search
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

    # Set up 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Use GridSearchCV to search for the best C according to accuracy.
    grid_search = GridSearchCV(lr, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
    grid_search.fit(X_scaled, y)

    # Output the best hyperparameters and corresponding cross-validated performance
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    # Now, you can evaluate the best model on your test set.
    # For demonstration, you might split the data chronologically (e.g., 80% train, 20% test).
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Refit the best model on the training split
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate predictions on the test set
    y_pred = best_model.predict(X_test)
    print("model coeffs", best_model.coef_)
    print("intercept", best_model.intercept_)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
