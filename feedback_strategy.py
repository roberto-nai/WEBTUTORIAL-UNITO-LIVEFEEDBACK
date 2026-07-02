"""
feedback_strategy.py

This module converts low-level behavioural indicators and prediction results into
an interpreted feedback context for the language model.

The main idea is to keep the decision logic outside the LLM. Process Mining and
prediction outputs are first summarised into student-facing behavioural evidence;
then the feedback intent is selected from the external JSON configuration file.
The LLM receives only the interpreted context, so it can generate feedback without
being asked to reason directly on technical metrics such as Cyclomatic Complexity
or event-log features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# The JSON file keeps the feedback-intent definitions and thresholds outside the
# Python code, so the strategy can be adjusted without editing this module.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INTENTS_PATH = BASE_DIR / "prompts" / "feedback_intents.json"


def load_feedback_intents(path: Path = DEFAULT_INTENTS_PATH) -> dict[str, Any]:
    """Load feedback intents, quiz-rate levels and behavioural thresholds."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_quiz_rate(value: float | None, config: dict[str, Any]) -> str:
    """Classify quiz accuracy into the Low/Medium/High levels used in the paper.

    The levels are read from the JSON configuration. This makes the thresholds
    explicit and aligned with the table reported in the paper.
    """
    # When no quiz information is available, the LLM should be informed rather
    # than receiving an artificial score.
    if value is None:
        return "Not available"

    # Each level can define inclusive/exclusive lower and upper bounds.
    levels = config.get("quiz_rate_levels", [])
    for level in levels:
        minimum = level.get("min")
        maximum = level.get("max")
        include_min = bool(level.get("include_min", True))
        include_max = bool(level.get("include_max", False))

        lower_ok = True if minimum is None else (value >= minimum if include_min else value > minimum)
        upper_ok = True if maximum is None else (value <= maximum if include_max else value < maximum)

        if lower_ok and upper_ok:
            label = level.get("label", "Unknown")
            description = level.get("description")
            return f"{label} ({description})" if description else str(label)

    return f"Unknown ({value:.2f})"


# Helper used by the summary functions to read thresholds from the JSON file.
# Defaults are provided to keep the module robust if a threshold is missing.
def _threshold(config: dict[str, Any], group: str, key: str, default: float) -> float:
    return float(config.get("behaviour_thresholds", {}).get(group, {}).get(key, default))


def summarise_engagement(features: dict[str, Any], config: dict[str, Any]) -> str:
    """Summarise how actively the student interacted with the tutorial."""
    pages = max(int(features.get("n_pages_prefix") or features.get("n_pages") or 1), 1)
    clicks = int(features.get("n_clicks", 0) or 0)
    double_clicks = int(features.get("n_dbclicks", 0) or 0)
    events = int(features.get("n_events_prefix") or features.get("n_events") or 0)

    # Interaction is normalised by the number of completed pages so that longer
    # and shorter prefixes can be compared more fairly.
    interaction_score = (clicks + double_clicks) / pages
    events_per_page = events / pages

    low_interaction = _threshold(config, "interaction_per_page", "low", 2)
    high_interaction = _threshold(config, "interaction_per_page", "high", 8)
    low_events = _threshold(config, "events_per_page", "low", 4)
    high_events = _threshold(config, "events_per_page", "high", 12)

    # The returned text is intentionally student-facing and avoids raw metrics.
    if interaction_score >= high_interaction or events_per_page >= high_events:
        return "High engagement, with active interaction during the learning session"
    if interaction_score < low_interaction and events_per_page < low_events:
        return "Low engagement, with limited interaction during the learning session"
    return "Moderate engagement, with some interaction during the learning session"


def summarise_navigation(features: dict[str, Any], config: dict[str, Any]) -> str:
    """Summarise the student's navigation behaviour across the tutorial."""
    backward_jumps = int(features.get("backward_jumps_prefix") or features.get("backward_jumps") or 0)
    forward_jumps = int(features.get("forward_jumps_prefix") or features.get("forward_jumps") or 0)
    complexity = int(features.get("cyclomatic_complexity_prefix") or features.get("cyclomatic_complexity") or 0)

    reflective_jumps = _threshold(config, "backward_jumps", "reflective", 1)
    frequent_jumps = _threshold(config, "backward_jumps", "frequent", 3)
    frequent_forward_jumps = _threshold(config, "forward_jumps", "frequent", 2)
    high_complexity = _threshold(config, "cyclomatic_complexity", "high", 4)

    # Frequent jumps or high process complexity suggest a non-linear path. A few
    # backward jumps can instead indicate reflective navigation.
    if backward_jumps >= frequent_jumps or forward_jumps >= frequent_forward_jumps or complexity >= high_complexity:
        return "Complex navigation, with frequent forward/backward jumps or non-linear movements"
    if backward_jumps >= reflective_jumps:
        return "Reflective navigation, with some backward jumps to previous content"
    return "Mostly linear navigation through the learning content"


def summarise_time_management(features: dict[str, Any], config: dict[str, Any]) -> str:
    """Summarise how the student distributed time across learning activities."""
    avg_time = float(features.get("avg_time_per_page_prefix") or 0)
    total_time = int(features.get("total_time_seconds_prefix") or features.get("total_time_seconds") or 0)
    cv = float(features.get("coefficient_of_variation_prefix") or features.get("coefficient_of_variation") or 0.0)

    very_short = _threshold(config, "avg_time_per_page_seconds", "very_short", 30)
    adequate = _threshold(config, "avg_time_per_page_seconds", "adequate", 60)
    long = _threshold(config, "avg_time_per_page_seconds", "long", 180)
    irregular_cv = _threshold(config, "coefficient_of_variation", "irregular", 0.75)

    # Average time per page captures speed, while the coefficient of variation
    # captures irregularity in how time is distributed across activities.
    if total_time <= 0:
        return "Time information is not available"
    if avg_time < very_short:
        return "Very short time spent on activities"
    if avg_time < adequate:
        return "Limited time spent on activities"
    if cv >= irregular_cv:
        return "Irregular time spent across activities"
    if avg_time > long:
        return "Long time spent on activities"
    return "Adequate time spent on activities"


def get_intent_definition(predicted_outcome: str | None, config: dict[str, Any]) -> dict[str, Any]:
    """Return the feedback-intent definition associated with the predicted outcome."""
    intents = config.get("feedback_intents", {})
    fallback = config.get("default_predicted_outcome", "Medium")
    # If the classifier does not return one of the configured classes, fall back
    # to the default intent, usually the medium-risk Review Recommendation.
    outcome = predicted_outcome if predicted_outcome in intents else fallback
    return intents.get(outcome, intents.get("Medium", {}))


def build_feedback_context(
    *,
    features: dict[str, Any],
    prediction: dict[str, Any],
    config_path: Path = DEFAULT_INTENTS_PATH,
) -> dict[str, Any]:
    """Build the interpreted context passed to the LLM prompt.

    The LLM receives a feedback intent and a behavioural summary, not raw
    Process Mining metrics. This keeps the strategy selection transparent and
    outside the language model.
    """
    # Load the external strategy configuration for every call. This is simple and
    # makes changes to the JSON file immediately visible during development.
    config = load_feedback_intents(config_path)
    predicted_outcome = prediction.get("predicted_class")
    intent_definition = get_intent_definition(predicted_outcome, config)

    # Prefer prefix-level quiz accuracy when available, because feedback is
    # generated during the learning session. Fall back to the complete-session
    # value for offline or retrospective analyses.
    quiz_accuracy = features.get("quiz_accuracy_prefix")
    if quiz_accuracy is None:
        quiz_accuracy = features.get("quiz_accuracy")

    # This dictionary is the only information that the prompt needs: selected
    # intent, purpose, predicted outcome, and readable behavioural summaries.
    return {
        "feedback_intent": intent_definition.get("intent", "Review Recommendation"),
        "feedback_purpose": intent_definition.get("purpose", "Recommend an appropriate next action."),
        "behavioural_evidence": intent_definition.get("behavioural_evidence", "Mixed behavioural patterns."),
        "predicted_outcome": predicted_outcome or "Not available",
        "engagement_level": summarise_engagement(features, config),
        "quiz_rate": classify_quiz_rate(quiz_accuracy, config),
        "navigation_behaviour": summarise_navigation(features, config),
        "time_management": summarise_time_management(features, config),
    }
