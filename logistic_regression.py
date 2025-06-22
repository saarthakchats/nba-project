#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("data/games_2022_23_enriched_rolling.csv", parse_dates=["GAME_DATE"])
    # Drop any rows where our key features are missing (just in case)
    df = df.dropna(subset=["home_rolling_ENR", "away_rolling_ENR", "home_win"])

    # Sort the data in chronological order to maintain the time-series nature
    df = df.sort_values("GAME_DATE")
    # Create our feature matrix X and target vector y
    X = df[["home_rolling_ENR", "away_rolling_ENR"]].values
    y = df["home_win"].values

    # Perform a time-based train-test split (80% for training, 20% for testing)
    split_idx = int(0.8 * len(df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Standardize features so that they contribute equally to the regularization term
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train the logistic regression model
    # We use L2 regularization (the default in scikit-learn) with regularization parameter C=1.0.
    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Use the trained model to make predictions on the test set:
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate the performance using several metrics:
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("Test Accuracy: {:.2f}%".format(accuracy * 100))
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", conf_mat)
    print("ROC-AUC Score: {:.2f}".format(roc_auc))

    # Optionally, plot the ROC curve to visualize performance:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc))
    plt.plot([1],[1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Logistic Regression")
    plt.legend(loc="best")
    plt.show()
