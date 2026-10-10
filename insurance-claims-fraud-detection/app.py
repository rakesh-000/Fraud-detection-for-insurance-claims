from pathlib import Path
from datetime import datetime, timezone
import csv, json, os
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "predictions.csv"

app = Flask(__name__)
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
FEATURES = [
    "claim_amount", "policy_age_days", "claim_delay_days",
    "incident_hour", "num_previous_claims", "vehicle_age",
    "claim_type", "incident_location", "police_report", "witness_present"
]
NUMERIC = ["claim_amount", "policy_age_days", "claim_delay_days", "incident_hour", "num_previous_claims", "vehicle_age"]

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def log_prediction(result):
    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "prediction", "fraud_probability", "risk_level", "threshold"
        ])
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": result["timestamp"],
            "prediction": result["prediction"],
            "fraud_probability": result["fraud_probability"],
            "risk_level": result["risk_level"],
            "threshold": result["threshold"]
        })

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})

@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "Metrics not found. Run python train.py first."}), 404
    return jsonify(json.loads(METRICS_PATH.read_text(encoding="utf-8")))

@app.get("/api/history")
def history():
    if not LOG_PATH.exists():
        return jsonify([])
    df = pd.read_csv(LOG_PATH).tail(50).iloc[::-1]
    return jsonify(df.fillna("").to_dict(orient="records"))

@app.post("/api/threshold")
def update_threshold():
    global THRESHOLD
    body = request.get_json(silent=True) or {}
    try:
        value = float(body.get("threshold"))
    except (TypeError, ValueError):
        return jsonify({"error": "Provide a numeric threshold between 0 and 1."}), 400
    if not 0 <= value <= 1:
        return jsonify({"error": "Threshold must be between 0 and 1."}), 400
    THRESHOLD = value
    return jsonify({"threshold": THRESHOLD, "message": "Threshold updated for this running session."})

@app.post("/api/predict")
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Trained model not found. Run python train.py first."}), 503
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    missing = [name for name in FEATURES if name not in body]
    if missing:
        return jsonify({"error": "Missing required fields.", "missing_fields": missing}), 400
    try:
        row = {}
        for name in NUMERIC:
            row[name] = float(body[name])
        for name in FEATURES:
            if name not in NUMERIC:
                row[name] = str(body[name])
        X = pd.DataFrame([row], columns=FEATURES)
        probability = float(model.predict_proba(X)[0][1])
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400
    is_fraud = probability >= THRESHOLD
    risk = "High" if probability >= max(THRESHOLD, 0.7) else ("Medium" if probability >= 0.35 else "Low")
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": "Potential Fraud" if is_fraud else "Likely Genuine",
        "fraud_probability": round(probability, 4),
        "risk_level": risk,
        "threshold": THRESHOLD,
        "note": "Screening result only; human review is required."
    }
    log_prediction(result)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
