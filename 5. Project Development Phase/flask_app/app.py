"""
Credit Card Fraud Detection - Flask Web Application
-----------------------------------------------------
Loads the best trained model (from ../model/train_model.py output) and
serves:
  - GET  /                a form for entering a single transaction's features
  - POST /predict          returns fraud / genuine prediction for that transaction
  - POST /batch-predict    accepts a CSV upload and returns predictions for each row

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLUMNS = json.load(f)

FRAUD_THRESHOLD = 0.5  # configurable decision threshold


def prepare_row(row_dict):
    """Take a dict of raw feature values, scale Time/Amount, return ordered df row."""
    df = pd.DataFrame([row_dict])
    df = df[FEATURE_COLUMNS]  # enforce column order
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    return df


@app.route("/")
def home():
    return render_template("index.html", feature_columns=FEATURE_COLUMNS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row_dict = {col: float(request.form.get(col, 0)) for col in FEATURE_COLUMNS}
        X = prepare_row(row_dict)
        proba = float(model.predict_proba(X)[0][1])
        is_fraud = proba >= FRAUD_THRESHOLD
        result = {
            "prediction": "FRAUD" if is_fraud else "GENUINE",
            "fraud_probability": round(proba, 4),
        }
        return render_template("index.html", feature_columns=FEATURE_COLUMNS, result=result)
    except Exception as e:
        return render_template("index.html", feature_columns=FEATURE_COLUMNS,
                                error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint, e.g. for programmatic / IBM Watson-style calls."""
    data = request.get_json(force=True)
    X = prepare_row(data)
    proba = float(model.predict_proba(X)[0][1])
    return jsonify({
        "prediction": "FRAUD" if proba >= FRAUD_THRESHOLD else "GENUINE",
        "fraud_probability": round(proba, 4),
    })


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    file = request.files["file"]
    df = pd.read_csv(file)
    X = df[FEATURE_COLUMNS].copy()
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    proba = model.predict_proba(X)[:, 1]
    df["fraud_probability"] = proba.round(4)
    df["prediction"] = ["FRAUD" if p >= FRAUD_THRESHOLD else "GENUINE" for p in proba]
    return df.to_html(classes="table table-striped", index=False)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
