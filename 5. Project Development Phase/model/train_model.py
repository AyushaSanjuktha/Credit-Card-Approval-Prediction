"""
Credit Card Fraud Detection - Model Training Pipeline
------------------------------------------------------
Trains and compares 4 classifiers (Logistic Regression, Decision Tree,
Random Forest, XGBoost) on the ULB Credit Card Fraud dataset
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), handles the
extreme class imbalance with SMOTE, and saves the best-performing model
for deployment in the Flask app / IBM Watson ML.

Usage:
    1. Download creditcard.csv from the Kaggle link above and place it
       in this same folder (model/creditcard.csv).
    2. Run: python train_model.py
    3. Outputs: best_model.pkl, scaler.pkl, model_comparison.csv
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

DATA_PATH = "creditcard.csv"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} transactions | Fraud cases: {df['Class'].sum()} "
          f"({df['Class'].mean()*100:.3f}%)")
    return df


def preprocess(df):
    """Scale Time & Amount (the only non-PCA columns); leave V1-V28 as is."""
    scaler = StandardScaler()
    df = df.copy()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y, scaler


def train_and_compare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Address extreme class imbalance (~0.17% fraud) with SMOTE on the
    # training split only (test split stays untouched / realistic).
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_bal):,} training rows "
          f"({y_train_bal.sum():,} fraud / {len(y_train_bal)-y_train_bal.sum():,} genuine)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4),
            "AUPRC": round(average_precision_score(y_test, y_proba), 4),
        }
        results.append(metrics)
        fitted[name] = model
        print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values("AUPRC", ascending=False)
    print("\n=== Model Comparison (sorted by AUPRC — best metric for rare-fraud) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")
    return best_name, best_model, results_df, X_test, y_test


def main():
    df = load_data()
    X, y, scaler = preprocess(df)
    best_name, best_model, results_df, X_test, y_test = train_and_compare(X, y)

    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    results_df.to_csv("model_comparison.csv", index=False)
    with open("feature_columns.json", "w") as f:
        json.dump(list(X_test.columns), f)
    with open("best_model_name.txt", "w") as f:
        f.write(best_name)

    print("\nSaved: best_model.pkl, scaler.pkl, model_comparison.csv, "
          "feature_columns.json, best_model_name.txt")


if __name__ == "__main__":
    main()
