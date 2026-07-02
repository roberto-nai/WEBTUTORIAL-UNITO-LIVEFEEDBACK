"""
    app.py
    Live Help Flask application for real-time session analysis and feedback generation.
    This application provides endpoints to visualize session events, generate process-aware feedback, and serve DFG visualizations using PM4Py.
    It also includes API endpoints for training and debugging the XGBoost model used for predictions.
"""
from __future__ import annotations

import os
from statistics import median
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, send_from_directory, url_for

from llm_service import generate_llm_feedback
from log_service import calculate_cyclomatic_complexity, create_dfg_png, export_session_event_log_csv
from ml_service import add_quiz_features, build_prefix_quiz, extract_basic_features, run_xgb_prediction, train_xgb_model
from sql_service import (
    get_training_sessions_debug_stats,
    load_interaction_features,
    load_session_event_log,
    load_session_events,
    load_session_quiz,
)


BASE_DIR = Path(__file__).resolve().parent

REFRESH_SECONDS = int(os.getenv("HELP_REFRESH_SECONDS", "30"))

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

DFG_DIR = BASE_DIR / "dfg"
DFG_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def format_seconds(seconds: int) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    return f"{minutes}m {sec}s"


def format_mm_ss(seconds: float | int) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, sec = divmod(total_seconds, 60)
    return f"{minutes:02d}:{sec:02d}"


def parse_m_ss_to_seconds(value: str) -> int:
    raw = (value or "").strip()
    if not raw or ":" not in raw:
        return 0
    minute_part, second_part = raw.split(":", 1)
    try:
        minutes = int(minute_part)
        seconds = int(second_part)
    except ValueError:
        return 0
    return max(0, minutes * 60 + seconds)


def build_live_help_payload(session_id: str) -> dict[str, Any]:
    df = load_session_events(session_id)
    event_log_df = load_session_event_log(session_id)
    export_session_event_log_csv(event_log_df, session_id, LOGS_DIR)

    page_durations_seconds: list[int] = []
    if not event_log_df.empty:
        duration_rows = event_log_df.loc[
            event_log_df["show_duration"] & event_log_df["duration"].astype(str).ne(""),
            "duration",
        ]
        page_durations_seconds = [parse_m_ss_to_seconds(str(value)) for value in duration_rows]

    total_duration_seconds = int(sum(page_durations_seconds)) if page_durations_seconds else 0
    avg_duration_seconds = (total_duration_seconds / len(page_durations_seconds)) if page_durations_seconds else 0
    median_duration_seconds = median(page_durations_seconds) if page_durations_seconds else 0

    event_log_summary = {
        "total_duration": format_mm_ss(total_duration_seconds),
        "avg_per_page": format_mm_ss(avg_duration_seconds),
        "median_per_page": format_mm_ss(median_duration_seconds),
    }

    features = extract_basic_features(df)
    quiz_df = load_session_quiz(session_id)

    quiz_results: list[dict[str, Any]] = []
    quiz_summary = {
        "total": 0,
        "correct": 0,
        "wrong": 0,
        "correct_pct": 0.0,
        "wrong_pct": 0.0,
    }

    if not quiz_df.empty:
        quiz_view_df = quiz_df.copy()
        quiz_view_df["page"] = (
            quiz_view_df["pageTitle"]
            .fillna(quiz_view_df["pageName"])
            .fillna("Unknown page")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        quiz_view_df["answer"] = quiz_view_df["answer"].fillna("-").astype(str)
        quiz_view_df["result"] = quiz_view_df["answerCorrect"].apply(
            lambda value: "Correct" if int(value) == 1 else "Wrong"
        )
        quiz_view_df["timestamp"] = quiz_view_df["lastUpdate"]
        quiz_view_df.insert(0, "row", range(1, len(quiz_view_df) + 1))

        total = int(len(quiz_view_df))
        correct = int((quiz_view_df["answerCorrect"] == 1).sum())
        wrong = total - correct

        quiz_summary = {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "correct_pct": round((correct / total) * 100, 1) if total else 0.0,
            "wrong_pct": round((wrong / total) * 100, 1) if total else 0.0,
        }

        quiz_results = quiz_view_df[["row", "page", "answer", "result", "timestamp"]].to_dict(orient="records")

    prefix_quiz_df = build_prefix_quiz(quiz_df, prefix_pages=3)
    features = add_quiz_features(features, prefix_quiz_df)
    features.update(load_interaction_features(session_id))
    features["cyclomatic_complexity"] = calculate_cyclomatic_complexity(df)
    features["total_time_readable"] = format_seconds(features["total_time_seconds"])

    prediction = run_xgb_prediction(features)
    llm = generate_llm_feedback(features, prediction, df)
    dfg_png_url = create_dfg_png(df, session_id, DFG_DIR)
    latest_events = [] if df.empty else df.tail(10)["activity"].tolist()

    return {
        "session_id": session_id,
        "features": features,
        "prediction": prediction,
        "llm": llm,
        "dfg_png_url": dfg_png_url,
        "event_log": event_log_df.to_dict(orient="records"),
        "event_log_summary": event_log_summary,
        "quiz_results": quiz_results,
        "quiz_summary": quiz_summary,
        "latest_events": latest_events,
    }


@app.route("/")

def home():
    return "Live Feedback is running. Use /live_help/<sessionID>."


@app.route("/<session_id>") # REDIRECT: Accept session IDs directly at root path for compatibility.
def live_help_root_session(session_id: str):
    """Compatibility route: accept session IDs directly at root path."""
    return redirect(url_for("live_help", session_id=session_id))


@app.route("/live_help/<session_id>")
def live_help(session_id: str):
    payload = build_live_help_payload(session_id)

    return render_template(
        "index.html",
        session_id=payload["session_id"],
        features=payload["features"],
        prediction=payload["prediction"],
        llm=payload["llm"],
        dfg_png_url=payload["dfg_png_url"],
        event_log=payload["event_log"],
        event_log_summary=payload["event_log_summary"],
        quiz_results=payload["quiz_results"],
        quiz_summary=payload["quiz_summary"],
        latest_events=payload["latest_events"],
        refresh_seconds=REFRESH_SECONDS,
    )

@app.route("/live-help/<session_id>")
def live_help_legacy(session_id: str):
    return redirect(url_for("live_help", session_id=session_id))



@app.route("/dfg/<path:filename>")
def serve_dfg(filename: str):
    """Serve DFG PNG files generated by PM4Py."""
    return send_from_directory(DFG_DIR, filename)


@app.route("/api/live_help/<session_id>")
def live_help_api(session_id: str):
    payload = build_live_help_payload(session_id)

    return jsonify(
        {
            "sessionID": payload["session_id"],
            "features": payload["features"],
            "prediction": payload["prediction"],
            "llm_feedback": payload["llm"],
        }
    )


@app.route("/api/train_xgb")
def train_xgb_api():
    result = train_xgb_model(prefix_pages=3)
    return jsonify(result)


@app.route("/api/debug_training_sessions")
def debug_training_sessions():
    return jsonify(get_training_sessions_debug_stats())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
