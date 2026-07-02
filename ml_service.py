from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from sql_service import get_completed_session_ids, load_session_events, load_session_quiz

try:
    import joblib
except ImportError:
    joblib = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

XGB_MODEL_PATH = Path(os.getenv("XGB_MODEL_PATH", MODELS_DIR / "xgb_model.joblib"))
XGB_FEATURES_PATH = Path(os.getenv("XGB_FEATURES_PATH", MODELS_DIR / "xgb_features.joblib"))
TRAINING_DATASET_PATH = MODELS_DIR / "training_dataset.csv"

PERFORMANCE_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High",
}


def calculate_quiz_accuracy(quiz_df: pd.DataFrame) -> float | None:
    """Calculate the ratio of correct quiz answers."""
    if quiz_df.empty:
        return None
    return float(quiz_df["answerCorrect"].mean())


def classify_final_quiz_accuracy(quiz_accuracy: float) -> int:
    """Map final quiz accuracy to a three-class performance label."""
    if quiz_accuracy < 0.5:
        return 0
    if quiz_accuracy <= 0.8:
        return 1
    return 2


def add_quiz_features(features: dict[str, Any], quiz_df: pd.DataFrame) -> dict[str, Any]:
    """Add quiz-based features to an existing feature dictionary."""
    quiz_accuracy = calculate_quiz_accuracy(quiz_df)
    quiz_correct = int(quiz_df["answerCorrect"].sum()) if not quiz_df.empty else 0
    quiz_wrong = int(len(quiz_df) - quiz_correct)
    features["n_quiz"] = int(len(quiz_df))
    features["quiz_correct"] = quiz_correct
    features["quiz_wrong"] = quiz_wrong
    features["quiz_accuracy"] = quiz_accuracy
    features["n_quiz_prefix"] = int(len(quiz_df))
    features["quiz_correct_prefix"] = quiz_correct
    features["quiz_wrong_prefix"] = quiz_wrong
    features["quiz_accuracy_prefix"] = 0.0 if quiz_accuracy is None else quiz_accuracy
    return features


def calculate_cyclomatic_complexity(df: pd.DataFrame) -> int:
    """Calculate the cyclomatic complexity of the current page-level DFG."""
    if df.empty:
        return 0

    activities = df["concept:name"].dropna().astype(str).tolist()
    if not activities:
        return 0

    trace = ["START", *activities, "END"]
    nodes = set(trace)
    edges = set(zip(trace[:-1], trace[1:]))

    complexity = len(edges) - len(nodes) + 2
    return max(1, int(complexity))


def calculate_coefficient_of_variation(df: pd.DataFrame) -> float:
    """Calculate the coefficient of variation of time spent across visited pages."""
    if df.empty or "lastUpdate" not in df.columns:
        return 0.0

    working_df = df.copy()
    working_df["pageOrderNumeric"] = normalise_page_order(working_df["pageOrder"])
    working_df = working_df.dropna(subset=["pageOrderNumeric", "lastUpdate"])

    if working_df.empty:
        return 0.0

    page_durations = (
        working_df.groupby("pageOrderNumeric")["lastUpdate"]
        .agg(lambda values: (values.max() - values.min()).total_seconds())
        .astype(float)
    )
    page_durations = page_durations[page_durations > 0]

    if len(page_durations) < 2 or page_durations.mean() <= 0:
        return 0.0

    return float(page_durations.std(ddof=0) / page_durations.mean())


def normalise_page_order(series: pd.Series) -> pd.Series:
    """Convert pageOrder values such as '01' and '010' to numeric values."""
    return pd.to_numeric(series.astype(str).str.replace(r"\D", "", regex=True), errors="coerce")


def build_prefix_trace(df: pd.DataFrame, prefix_pages: int = 3) -> pd.DataFrame:
    """Keep only the events belonging to the first prefix_pages pages."""
    if df.empty:
        return df

    df = df.copy()
    df["pageOrderNumeric"] = normalise_page_order(df["pageOrder"])
    return df[df["pageOrderNumeric"] <= prefix_pages].copy()


def build_prefix_quiz(quiz_df: pd.DataFrame, prefix_pages: int = 3) -> pd.DataFrame:
    """Keep only quiz answers belonging to the first prefix_pages pages."""
    if quiz_df.empty:
        return quiz_df

    quiz_df = quiz_df.copy()
    quiz_df["pageOrderNumeric"] = normalise_page_order(quiz_df["pageOrder"])
    return quiz_df[quiz_df["pageOrderNumeric"] <= prefix_pages].copy()


def extract_basic_features(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "n_events": 0,
            "n_pages": 0,
            "n_quiz": 0,
            "quiz_correct": 0,
            "quiz_wrong": 0,
            "quiz_accuracy": None,
            "total_time_seconds": 0,
            "backward_jumps": 0,
            "cyclomatic_complexity": 0,
            "coefficient_of_variation": 0.0,
            "forward_jumps": 0,
            "n_events_prefix": 0,
            "n_pages_prefix": 0,
            "total_time_seconds_prefix": 0,
            "avg_time_per_page_prefix": 0,
            "backward_jumps_prefix": 0,
            "cyclomatic_complexity_prefix": 0,
            "coefficient_of_variation_prefix": 0.0,
            "forward_jumps_prefix": 0,
            "n_quiz_prefix": 0,
            "quiz_correct_prefix": 0,
            "quiz_wrong_prefix": 0,
            "quiz_accuracy_prefix": 0.0,
        }

    page_orders = normalise_page_order(df["pageOrder"])
    page_order_diff = page_orders.diff()
    backward_jumps = int((page_order_diff < 0).sum())
    forward_jumps = int((page_order_diff > 1).sum())

    total_time_seconds = 0
    if len(df) >= 2:
        total_time_seconds = int((df["lastUpdate"].max() - df["lastUpdate"].min()).total_seconds())

    cyclomatic_complexity = calculate_cyclomatic_complexity(df)
    coefficient_of_variation = calculate_coefficient_of_variation(df)
    n_pages = int(normalise_page_order(df[df["source_table"] == "page_event"]["pageOrder"]).nunique())
    avg_time_per_page = total_time_seconds / max(n_pages, 1)

    return {
        "n_events": int(len(df)),
        "n_pages": n_pages,
        "n_quiz": 0,
        "quiz_correct": 0,
        "quiz_wrong": 0,
        "quiz_accuracy": None,
        "total_time_seconds": total_time_seconds,
        "backward_jumps": backward_jumps,
        "cyclomatic_complexity": cyclomatic_complexity,
        "coefficient_of_variation": coefficient_of_variation,
        "forward_jumps": forward_jumps,
        "n_events_prefix": int(len(df)),
        "n_pages_prefix": n_pages,
        "total_time_seconds_prefix": total_time_seconds,
        "avg_time_per_page_prefix": avg_time_per_page,
        "backward_jumps_prefix": backward_jumps,
        "cyclomatic_complexity_prefix": cyclomatic_complexity,
        "coefficient_of_variation_prefix": coefficient_of_variation,
        "forward_jumps_prefix": forward_jumps,
        "n_quiz_prefix": 0,
        "quiz_correct_prefix": 0,
        "quiz_wrong_prefix": 0,
        "quiz_accuracy_prefix": 0.0,
    }


def extract_prediction_features_from_session(
    events_df: pd.DataFrame,
    quiz_df: pd.DataFrame,
    prefix_pages: int = 3,
) -> dict[str, Any]:
    """Extract prefix-based features for XGB prediction."""
    prefix_df = build_prefix_trace(events_df, prefix_pages=prefix_pages)
    prefix_quiz_df = build_prefix_quiz(quiz_df, prefix_pages=prefix_pages)

    features = extract_basic_features(prefix_df)
    features = add_quiz_features(features, prefix_quiz_df)

    n_pages = max(features["n_pages"], 1)
    features["n_events_prefix"] = features["n_events"]
    features["n_pages_prefix"] = features["n_pages"]
    features["total_time_seconds_prefix"] = features["total_time_seconds"]
    features["avg_time_per_page_prefix"] = features["total_time_seconds"] / n_pages
    features["backward_jumps_prefix"] = features["backward_jumps"]
    features["cyclomatic_complexity_prefix"] = features["cyclomatic_complexity"]
    features["coefficient_of_variation_prefix"] = features["coefficient_of_variation"]
    features["forward_jumps_prefix"] = features["forward_jumps"]

    return features


def build_training_dataset(
    prefix_pages: int = 3,
    expected_pages: int = 10,
    expected_quizzes: int = 10,
) -> pd.DataFrame:
    """Build the XGB training dataset from complete historical sessions."""
    rows = []
    session_ids = get_completed_session_ids(
        expected_pages=expected_pages,
        expected_quizzes=expected_quizzes,
    )

    for session_id in session_ids:
        events_df = load_session_events(session_id)
        quiz_df = load_session_quiz(session_id)

        final_quiz_accuracy = calculate_quiz_accuracy(quiz_df)
        if final_quiz_accuracy is None:
            continue

        target_class = classify_final_quiz_accuracy(final_quiz_accuracy)

        features = extract_prediction_features_from_session(
            events_df=events_df,
            quiz_df=quiz_df,
            prefix_pages=prefix_pages,
        )

        row = {
            "sessionID": session_id,
            "target_class": target_class,
            "target_label": PERFORMANCE_LABELS[target_class],
            "final_quiz_accuracy": final_quiz_accuracy,
            "n_events_prefix": features["n_events_prefix"],
            "n_pages_prefix": features["n_pages_prefix"],
            "total_time_seconds_prefix": features["total_time_seconds_prefix"],
            "avg_time_per_page_prefix": features["avg_time_per_page_prefix"],
            "backward_jumps_prefix": features["backward_jumps_prefix"],
            "cyclomatic_complexity_prefix": features["cyclomatic_complexity_prefix"],
            "n_quiz_prefix": features["n_quiz_prefix"],
            "quiz_accuracy_prefix": features["quiz_accuracy_prefix"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def train_xgb_model(prefix_pages: int = 3) -> dict[str, Any]:
    """Train and save a three-class XGB model on complete historical sessions."""
    if joblib is None:
        return {"status": "error", "message": "joblib is not installed."}

    if XGBClassifier is None:
        return {"status": "error", "message": "xgboost is not installed."}

    training_df = build_training_dataset(prefix_pages=prefix_pages)

    if training_df.empty:
        return {
            "status": "error",
            "message": "No complete sessions found for training.",
            "n_distinct_cases": 0,
            "label_distribution": {},
        }

    feature_names = [
        "n_events_prefix",
        "n_pages_prefix",
        "total_time_seconds_prefix",
        "avg_time_per_page_prefix",
        "backward_jumps_prefix",
        "cyclomatic_complexity_prefix",
        "n_quiz_prefix",
        "quiz_accuracy_prefix",
    ]

    n_distinct_cases = int(training_df["sessionID"].nunique())
    label_distribution = {
        PERFORMANCE_LABELS[int(label)]: int(count)
        for label, count in training_df["target_class"].value_counts().sort_index().items()
    }

    if training_df["target_class"].nunique() < 2:
        training_df.to_csv(TRAINING_DATASET_PATH, index=False)
        return {
            "status": "error",
            "message": "Training requires at least two performance classes.",
            "n_sessions": int(len(training_df)),
            "n_distinct_cases": n_distinct_cases,
            "label_distribution": label_distribution,
            "training_dataset": str(TRAINING_DATASET_PATH),
        }

    X = training_df[feature_names]
    y = training_df["target_class"]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X, y)

    joblib.dump(model, XGB_MODEL_PATH)
    joblib.dump(feature_names, XGB_FEATURES_PATH)
    training_df.to_csv(TRAINING_DATASET_PATH, index=False)

    return {
        "status": "ready",
        "message": "Three-class XGB model trained and saved.",
        "target_definition": {
            "Low": "final quiz accuracy < 0.50",
            "Medium": "0.50 <= final quiz accuracy <= 0.80",
            "High": "final quiz accuracy > 0.80",
        },
        "n_sessions": int(len(training_df)),
        "n_distinct_cases": n_distinct_cases,
        "label_distribution": label_distribution,
        "features": feature_names,
        "model_path": str(XGB_MODEL_PATH),
        "features_path": str(XGB_FEATURES_PATH),
        "training_dataset": str(TRAINING_DATASET_PATH),
    }


def format_seconds(seconds: int) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"


def run_xgb_prediction(features: dict[str, Any]) -> dict[str, Any]:
    """Run three-class XGB prediction after the first 3 pages."""
    if features["n_pages_prefix"] < 3:
        return {
            "status": "waiting",
            "message": "Prediction will start after the first 3 pages.",
            "predicted_class": None,
            "confidence": None,
            "probabilities": None,
        }

    if joblib is None or not XGB_MODEL_PATH.exists() or not XGB_FEATURES_PATH.exists():
        return {
            "status": "placeholder",
            "message": "XGB model not loaded yet. Train it on historical sessions and save it in live_help/models/.",
            "predicted_class": None,
            "confidence": None,
            "probabilities": None,
        }

    model = joblib.load(XGB_MODEL_PATH)
    feature_names = joblib.load(XGB_FEATURES_PATH)
    row = pd.DataFrame([{name: features.get(name, 0) for name in feature_names}])

    probabilities_raw = model.predict_proba(row)[0]
    predicted_class_id = int(probabilities_raw.argmax())

    probabilities = {
        PERFORMANCE_LABELS[class_id]: float(probabilities_raw[class_id])
        for class_id in sorted(PERFORMANCE_LABELS)
    }

    return {
        "status": "ready",
        "message": "Three-class prediction computed from the current session prefix.",
        "predicted_class": PERFORMANCE_LABELS[predicted_class_id],
        "predicted_class_id": predicted_class_id,
        "confidence": float(probabilities_raw[predicted_class_id]),
        "probabilities": probabilities,
    }
