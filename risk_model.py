"""
risk_model.py
Step 4: Risk Prediction.

A small, explainable Decision Tree (scikit-learn) trained on the six
features called out in the design doc:
    transaction_count, total_amount, linked_accounts,
    linked_phone_numbers, sim_changes, transaction_velocity

When a labeled investigation dataset is available, pass it to
`load_or_train_model()` to train and persist a case-specific model. The
synthetic dataset remains only as a fallback for first-run demonstrations.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

FEATURES = [
    "transaction_count", "total_amount", "linked_accounts",
    "linked_phone_numbers", "sim_changes", "transaction_velocity",
]

MODEL_PATH = "output/risk_model.joblib"


def load_training_data(filepath: str) -> pd.DataFrame:
    """Read a labeled feature dataset from CSV or Excel."""
    extension = os.path.splitext(filepath)[1].lower()
    if extension == ".csv":
        training_df = pd.read_csv(filepath)
    elif extension in (".xlsx", ".xls"):
        training_df = pd.read_excel(filepath)
    else:
        raise ValueError("Labeled training data must be a CSV or Excel file.")

    training_df.columns = [str(column).strip().lower() for column in training_df.columns]
    return training_df


def _synthetic_label(row) -> int:
    """Heuristic ground truth used only to bootstrap the demo model."""
    score = 0
    if row["linked_accounts"] >= 5:
        score += 1
    if row["transaction_velocity"] >= 3:
        score += 1
    if row["sim_changes"] >= 1:
        score += 1
    if row["transaction_count"] >= 10:
        score += 1
    return 1 if score >= 2 else 0


def generate_synthetic_training_data(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "transaction_count": rng.poisson(4, n),
        "total_amount": rng.exponential(15000, n).round(2),
        "linked_accounts": rng.poisson(2, n),
        "linked_phone_numbers": rng.poisson(1, n),
        "sim_changes": rng.poisson(0.3, n),
        "transaction_velocity": rng.exponential(1.2, n).round(3),
    })
    df["is_fraud"] = df.apply(_synthetic_label, axis=1)
    return df


def train_model(
    training_df: pd.DataFrame = None,
    save: bool = True,
    training_source: str = "synthetic_demo",
) -> DecisionTreeClassifier:
    if training_df is None:
        training_df = generate_synthetic_training_data()

    missing = [column for column in FEATURES + ["is_fraud"] if column not in training_df.columns]
    if missing:
        raise ValueError(
            "Training data is missing required columns: " + ", ".join(missing)
        )

    training_df = training_df[FEATURES + ["is_fraud"]].copy()
    training_df[FEATURES] = training_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    training_df["is_fraud"] = pd.to_numeric(training_df["is_fraud"], errors="coerce")
    training_df = training_df.dropna().reset_index(drop=True)
    training_df["is_fraud"] = training_df["is_fraud"].astype(int)

    if len(training_df) < 10 or training_df["is_fraud"].nunique() < 2:
        raise ValueError(
            "Training data needs at least 10 valid rows with both 0 and 1 "
            "values in is_fraud."
        )

    X = training_df[FEATURES]
    y = training_df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    report["status"] = "trained_model"
    report["training_source"] = training_source
    report["training_rows"] = len(training_df)

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

    return model, report


def load_or_train_model(training_df: pd.DataFrame = None, training_source: str = None):
    if training_df is not None:
        return train_model(
            training_df,
            training_source=training_source or "uploaded_labeled_data",
        )

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        report = {
            "status": "loaded_existing_model",
            "model_path": MODEL_PATH,
            "training_source": "persisted_model",
            "note": "Loaded the persisted Decision Tree model from disk. Upload labeled investigation data to retrain it.",
        }
        return model, report

    model, report = train_model(training_source="synthetic_demo")
    return model, report


def score_entities(entity_features: pd.DataFrame, model: DecisionTreeClassifier = None) -> pd.DataFrame:
    """Adds risk_score (0-100) and risk_level (High/Medium/Low) columns."""
    if model is None:
        model, _ = load_or_train_model()

    df = entity_features.copy()
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    if df.empty:
        df["risk_score"] = pd.Series(dtype=float)
        df["risk_level"] = pd.Series(dtype=object)
        return df

    X = df[FEATURES].fillna(0)
    proba = model.predict_proba(X)[:, 1]  # probability of "fraud" class
    df["risk_score"] = (proba * 100).round(1)

    def level(score):
        if score >= 75:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    df["risk_level"] = df["risk_score"].map(level)
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)
