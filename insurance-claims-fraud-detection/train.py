from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "claims.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "claim_amount", "policy_age_days", "claim_delay_days",
    "incident_hour", "num_previous_claims", "vehicle_age"
]
CATEGORICAL_FEATURES = [
    "claim_type", "incident_location", "police_report", "witness_present"
]
TARGET = "fraud"

def main():
    df = pd.read_csv(DATA_PATH)
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=[TARGET]).copy()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("Target column 'fraud' must contain both 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES)
    ])

    # Resampling is applied only to the training data. If SMOTE cannot be
    # used (e.g. very few minority examples), fall back to class weighting.
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        model = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=min(5, max(1, int(y_train.value_counts().min()) - 1)))),
            ("classifier", RandomForestClassifier(
                n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ])
        try:
            model.fit(X_train, y_train)
        except ValueError:
            model = SkPipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1
                ))
            ])
            model.fit(X_train, y_train)
    except ImportError:
        model = SkPipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ])
        model.fit(X_train, y_train)

    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "test_rows": int(len(y_test)),
        "positive_test_rows": int(y_test.sum()),
        "model": "RandomForestClassifier"
    }
    joblib.dump(model, MODEL_DIR / "fraud_pipeline.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {MODEL_DIR / 'fraud_pipeline.joblib'}")

if __name__ == "__main__":
    main()
