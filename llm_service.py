from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import ollama
import pandas as pd

from feedback_strategy import build_feedback_context


BASE_DIR = Path(__file__).resolve().parent
ENABLE_LOCAL_LLM = int(os.getenv("ENABLE_LOCAL_LLM", "1"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
PROMPT_PATH = BASE_DIR / "prompts" / "process_feedback_prompt_v2.json"

# Reuse the exact same feedback while the input signal is unchanged.
_FEEDBACK_CACHE: dict[str, dict[str, Any]] = {}
_MAX_CACHE_ITEMS = 256


def load_prompt_template(prompt_path: Path) -> dict[str, str]:
    with prompt_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_feedback_signature(
    *,
    feedback_context: dict[str, Any],
    visited_pages: list[str],
    model: str,
) -> str:
    payload = {
        "model": model,
        "feedback_context": feedback_context,
        "visited_pages": visited_pages,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sanitize_feedback_text(text: str) -> str:
    """Remove redundant lead-in phrases already implied by the UI section title."""
    cleaned = (text or "").strip()
    lead_in_patterns = [
        r"^here(?:'s| is) your personalised feedback\s*[:\-]\s*",
        r"^here(?:'s| is) your personali[sz]ed feedback\s*[:\-]\s*",
        r"^here(?:'s| is) some personalised feedback\s*[:\-]\s*",
        r"^here(?:'s| is) some personali[sz]ed feedback\s*[:\-]\s*",
        r"^personalised feedback\s*[:\-]\s*",
        r"^personali[sz]ed feedback\s*[:\-]\s*",
    ]

    for pattern in lead_in_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned


def generate_process_aware_feedback(
    *,
    feedback_context: dict[str, Any],
    visited_pages: list[str],
    prompt_path: Path,
    model: str = "llama3.2",
    temperature: float = 0.1,
    enabled: bool = False,
) -> dict[str, Any]:

    signature = _build_feedback_signature(
        feedback_context=feedback_context,
        visited_pages=visited_pages,
        model=model,
    )

    cached_feedback = _FEEDBACK_CACHE.get(signature)
    if cached_feedback is not None:
        return dict(cached_feedback)

    if not enabled:
        result = {
            "status": "disabled",
            "text": "Local LLM is disabled.",
            "feedback_intent": feedback_context.get("feedback_intent"),
            "predicted_outcome": feedback_context.get("predicted_outcome"),
            "behaviour_summary": feedback_context,
        }
        _FEEDBACK_CACHE[signature] = dict(result)
        return result

    prompt_template = load_prompt_template(prompt_path)

    user_prompt = prompt_template["user_template"].format(
        feedback_intent=feedback_context.get("feedback_intent", "Review Recommendation"),
        engagement_level=feedback_context.get("engagement_level", "Not available"),
        quiz_rate=feedback_context.get("quiz_rate", "Not available"),
        navigation_behaviour=feedback_context.get("navigation_behaviour", "Not available"),
        time_management=feedback_context.get("time_management", "Not available"),
        predicted_outcome=feedback_context.get("predicted_outcome", "Not available"),
        visited_pages="\n".join(f"- {page}" for page in visited_pages),
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": prompt_template["system"],
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": temperature,
                "top_p": 0.1,
                "num_predict": 180,
            },
        )

        result = {
            "status": "ready",
            "text": _sanitize_feedback_text(response["message"]["content"]),
            "feedback_intent": feedback_context.get("feedback_intent"),
            "predicted_outcome": feedback_context.get("predicted_outcome"),
            "behaviour_summary": feedback_context,
        }

        if len(_FEEDBACK_CACHE) >= _MAX_CACHE_ITEMS:
            _FEEDBACK_CACHE.clear()
        _FEEDBACK_CACHE[signature] = dict(result)
        return result

    except Exception as exc:
        return {
            "status": "error",
            "text": f"Local LLM error: {exc}",
            "feedback_intent": feedback_context.get("feedback_intent"),
            "predicted_outcome": feedback_context.get("predicted_outcome"),
            "behaviour_summary": feedback_context,
        }


def generate_llm_feedback(
    features: dict[str, Any],
    prediction: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Build interpreted feedback context and delegate to local LLM generation."""
    if prediction.get("status") != "ready":
        return {
            "status": prediction.get("status", "waiting"),
            "text": "Process-aware feedback will be generated after the learning outcome prediction is available.",
            "feedback_intent": None,
            "predicted_outcome": None,
            "behaviour_summary": None,
        }

    visited_pages = [] if df.empty else df["activity"].dropna().drop_duplicates().tolist()
    feedback_context = build_feedback_context(features=features, prediction=prediction)

    return generate_process_aware_feedback(
        feedback_context=feedback_context,
        visited_pages=visited_pages,
        prompt_path=PROMPT_PATH,
        model=OLLAMA_MODEL,
        temperature=0.0,
        enabled=ENABLE_LOCAL_LLM == 1,
    )
