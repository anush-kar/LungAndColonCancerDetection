from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS: List[str] = [
    "age",
    "gender",
    "air_pollution",
    "dust_allergy",
    "smoking",
    "passive_smoker",
]

TARGET_COLUMN = "level"

FEATURE_ALIASES: Dict[str, str] = {
    "age": "age",
    "gender": "gender",
    "air_pollution": "air_pollution",
    "air pollution": "air_pollution",
    "dust_allergy": "dust_allergy",
    "dust allergy": "dust_allergy",
    "smoking": "smoking",
    "passive_smoker": "passive_smoker",
    "passive smoker": "passive_smoker",
}


def _normalize_name(name: str) -> str:
    return "_".join(str(name).strip().lower().split())


def _load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [_normalize_name(c) for c in df.columns]

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    clean_df = df[required_columns].copy()
    clean_df = clean_df.dropna()

    for col in FEATURE_COLUMNS:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df.dropna()
    clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].astype(str)

    return clean_df


def _risk_percent_from_proba(class_labels: List[str], probabilities: List[float]) -> float:
    weights = {
        "low": 0.0,
        "medium": 0.5,
        "high": 1.0,
    }
    total = 0.0
    for label, prob in zip(class_labels, probabilities):
        total += weights.get(str(label).strip().lower(), 0.0) * float(prob)
    return round(total * 100.0, 2)


def train_reduced_risk_model(
    csv_path: str | Path,
    artifact_path: str | Path,
    random_state: int = 42,
) -> Dict[str, Any]:
    df = _load_dataset(csv_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    candidate_models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
    }

    evaluation_rows: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        acc = float(accuracy_score(y_test, y_pred))
        loss = float(log_loss(y_test, y_prob, labels=list(model.classes_)))

        evaluation_rows.append(
            {
                "model_name": model_name,
                "macro_f1": macro_f1,
                "accuracy": acc,
                "log_loss": loss,
            }
        )
        trained_models[model_name] = model

    evaluation_rows.sort(key=lambda row: (-row["macro_f1"], row["log_loss"]))
    best = evaluation_rows[0]

    best_name = best["model_name"]

    class_labels_by_model: Dict[str, List[str]] = {
        name: list(model.classes_)
        for name, model in trained_models.items()
    }

    artifact = {
        "default_model_name": best_name,
        "models": trained_models,
        "available_models": list(trained_models.keys()),
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": class_labels_by_model,
        "metrics": {
            "selected_model": best,
            "all_models": evaluation_rows,
        },
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    return artifact


def load_risk_model(artifact_path: str | Path) -> Dict[str, Any]:
    return joblib.load(artifact_path)


def _is_multimodel_artifact(artifact: Dict[str, Any]) -> bool:
    models = artifact.get("models")
    default_model_name = artifact.get("default_model_name")
    return isinstance(models, dict) and bool(models) and isinstance(default_model_name, str)


def ensure_risk_model(artifact_path: str | Path, csv_path: str | Path) -> Dict[str, Any]:
    artifact_path = Path(artifact_path)
    if artifact_path.exists():
        loaded = load_risk_model(artifact_path)
        if _is_multimodel_artifact(loaded):
            return loaded

        # Artifact exists but is legacy single-model format; retrain in new multi-model format.
        return train_reduced_risk_model(csv_path=csv_path, artifact_path=artifact_path)

    return train_reduced_risk_model(csv_path=csv_path, artifact_path=artifact_path)


def _normalize_model_name(name: str) -> str:
    return _normalize_name(name)


def _resolve_model_name(artifact: Dict[str, Any], requested_model_name: str | None) -> str:
    available_models: List[str] = artifact.get("available_models") or []
    if not available_models:
        # Backward compatibility with legacy artifact structure.
        return artifact.get("model_name", "random_forest")

    canonical_map = {
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "logistic_regression": "logistic_regression",
        "logistic": "logistic_regression",
        "log_reg": "logistic_regression",
        "lr": "logistic_regression",
    }

    if requested_model_name:
        key = canonical_map.get(_normalize_model_name(requested_model_name))
        if key and key in available_models:
            return key
        raise ValueError(
            f"Unknown model_name '{requested_model_name}'. Allowed values: {available_models}"
        )

    default_model_name = artifact.get("default_model_name")
    if isinstance(default_model_name, str) and default_model_name in available_models:
        return default_model_name

    return available_models[0]


def parse_payload(payload: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    normalized_payload: Dict[str, Any] = {}
    for key, value in payload.items():
        canonical_key = FEATURE_ALIASES.get(_normalize_name(key))
        if canonical_key:
            normalized_payload[canonical_key] = value

    missing = [f for f in FEATURE_COLUMNS if f not in normalized_payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    parsed: Dict[str, float] = {}
    for field in FEATURE_COLUMNS:
        try:
            parsed[field] = float(normalized_payload[field])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{field}' must be numeric.")

    return parsed


def predict_patient_likelihood(
    payload: Dict[str, Any],
    artifact: Dict[str, Any],
    model_name: str | None = None,
) -> Dict[str, Any]:
    features = parse_payload(payload)

    input_df = pd.DataFrame([features], columns=artifact["feature_columns"])

    selected_model_name = _resolve_model_name(artifact, model_name)

    # Multi-model artifact support.
    if isinstance(artifact.get("models"), dict):
        model = artifact["models"][selected_model_name]
        class_labels_map = artifact.get("class_labels", {})
        class_labels: List[str] = list(class_labels_map.get(selected_model_name, model.classes_))
    else:
        # Legacy artifact support.
        model = artifact["model"]
        class_labels = list(artifact.get("class_labels", model.classes_))
        selected_model_name = artifact.get("model_name", selected_model_name)

    probabilities = model.predict_proba(input_df)[0]
    pred_idx = int(probabilities.argmax())
    pred_level = class_labels[pred_idx]

    class_probabilities = {
        cls: round(float(prob) * 100.0, 2)
        for cls, prob in zip(class_labels, probabilities)
    }

    risk_percent = _risk_percent_from_proba(class_labels, list(probabilities))

    return {
        "predicted_level": pred_level,
        "risk_percent": risk_percent,
        "class_probabilities": class_probabilities,
        "model_name": selected_model_name,
        "input_features": features,
    }
