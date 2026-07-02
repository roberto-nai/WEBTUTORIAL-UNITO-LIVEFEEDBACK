from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pymysql
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "8889")),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

SQLALCHEMY_ENGINE = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4"
)


def get_connection():
    return pymysql.connect(**DB_CONFIG)


def read_sql(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read SQL through SQLAlchemy 2.x compatible connection."""
    with SQLALCHEMY_ENGINE.connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params or {})


def format_duration_m_ss(seconds: float | int) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes == 0:
        return f"00:{remaining_seconds:02d}"
    return f"{minutes}:{remaining_seconds:02d}"


def load_session_events(session_id: str) -> pd.DataFrame:
    """Load the live page-level trace for one sessionID. Quiz and survey events are ignored."""
    query = """
    SELECT
        'page_event' AS source_table,
        idEvent,
        projectID,
        sessionID,
        lang,
        pageName,
        pageTitle,
        menu,
        pageOrder,
        pagePara,
        event,
        duration,
        lastUpdate
    FROM events
    WHERE sessionID = :session_id
      AND event IN ('ingressoPagina', 'uscitaPagina')
      AND pageName LIKE 'page-%'
      AND pageName <> 'page-survey.php'
    ORDER BY lastUpdate ASC, idEvent ASC
    """

    df = read_sql(query, params={"session_id": session_id})

    if df.empty:
        return df

    df["lastUpdate"] = pd.to_datetime(df["lastUpdate"])
    df["activity"] = (
        df["pageTitle"]
        .fillna(df["pageName"])
        .fillna("Unknown page")
        .astype(str)
        .str.strip()
    )

    df["case:concept:name"] = df["sessionID"]
    df["concept:name"] = df["activity"]
    df["time:timestamp"] = df["lastUpdate"]

    return df


def load_session_event_log(session_id: str) -> pd.DataFrame:
    """Load the session event log with page entries, page exits, clicks and double-clicks."""
    query = """
    SELECT
        idEvent,
        projectID,
        sessionID,
        pageName,
        pageTitle,
        event,
        lastUpdate
    FROM events
    WHERE sessionID = :session_id
      AND event IN ('ingressoPagina', 'uscitaPagina', 'click', 'dbclick')
      AND pageName LIKE 'page-%'
      AND pageName <> 'page-survey.php'
    ORDER BY lastUpdate ASC, idEvent ASC
    """

    df = read_sql(query, params={"session_id": session_id})

    if df.empty:
        return pd.DataFrame(
            columns=["row", "page", "event", "timestamp", "duration", "page_rowspan", "show_page", "show_duration"]
        )

    df["lastUpdate"] = pd.to_datetime(df["lastUpdate"])
    df["page"] = (
        df["pageTitle"]
        .fillna(df["pageName"])
        .fillna("Unknown page")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["event"] = df["event"].replace(
        {
            "ingressoPagina": "Page entry",
            "uscitaPagina": "Page exit",
            "click": "Click",
            "dbclick": "Double-click",
        }
    )
    df["timestamp"] = df["lastUpdate"]
    df.insert(0, "row", range(1, len(df) + 1))

    df["page_group"] = df["page"].ne(df["page"].shift()).cumsum()
    df["page_rowspan"] = df.groupby("page_group")["page"].transform("size")
    df["show_page"] = df["page"].ne(df["page"].shift())
    df["show_duration"] = df["show_page"]

    durations = df.groupby("page_group").agg(
        group_start=("timestamp", "min"),
        group_end=("timestamp", "max"),
    )
    durations["duration"] = durations["group_end"] - durations["group_start"]

    df = df.merge(durations[["duration"]], left_on="page_group", right_index=True, how="left")
    df["duration"] = df["duration"].dt.total_seconds().fillna(0).apply(format_duration_m_ss)
    df.loc[~df["show_duration"], "duration"] = ""

    return df[["row", "page", "event", "timestamp", "duration", "page_rowspan", "show_page", "show_duration"]]


def load_session_quiz(session_id: str) -> pd.DataFrame:
    """Load quiz answers for one sessionID."""
    query = """
    SELECT
        idEvent,
        projectID,
        sessionID,
        lang,
        pageName,
        pageTitle,
        menu,
        pageOrder,
        answer,
        answerCorrect,
        lastUpdate
    FROM quiz
    WHERE sessionID = :session_id
    ORDER BY lastUpdate ASC, idEvent ASC
    """

    df = read_sql(query, params={"session_id": session_id})

    if df.empty:
        return df

    df["lastUpdate"] = pd.to_datetime(df["lastUpdate"])
    df["answerCorrect"] = pd.to_numeric(df["answerCorrect"], errors="coerce").fillna(0).astype(int)

    return df


def load_interaction_features(session_id: str) -> dict[str, int]:
    query = """
    SELECT event, COUNT(*) AS n
    FROM events
    WHERE sessionID = :session_id
      AND event IN ('click', 'dbclick')
    GROUP BY event
    """

    df = read_sql(query, {"session_id": session_id})

    counts = {
        "n_clicks": 0,
        "n_dbclicks": 0,
    }

    if df.empty:
        return counts

    for _, row in df.iterrows():
        if row["event"] == "click":
            counts["n_clicks"] = int(row["n"])
        elif row["event"] == "dbclick":
            counts["n_dbclicks"] = int(row["n"])

    return counts


def get_completed_session_ids(expected_pages: int = 10, expected_quizzes: int = 10) -> list[str]:
    """Return session IDs with all tutorial pages and all quizzes completed. Survey is ignored."""
    events_query = """
    SELECT sessionID
    FROM events
    WHERE event IN ('ingressoPagina', 'uscitaPagina')
      AND pageName LIKE 'page-%'
      AND pageName <> 'page-survey.php'
    GROUP BY sessionID
    HAVING COUNT(DISTINCT pageName) = :expected_pages
    """

    quiz_query = """
    SELECT sessionID
    FROM quiz
    GROUP BY sessionID
    HAVING COUNT(DISTINCT pageName) = :expected_quizzes
    """

    events_df = read_sql(events_query, {"expected_pages": expected_pages})
    quiz_df = read_sql(quiz_query, {"expected_quizzes": expected_quizzes})

    event_sessions = set(events_df["sessionID"].dropna().astype(str)) if not events_df.empty else set()
    quiz_sessions = set(quiz_df["sessionID"].dropna().astype(str)) if not quiz_df.empty else set()

    return sorted(event_sessions.intersection(quiz_sessions))


def get_training_sessions_debug_stats() -> dict[str, int]:
        """Return summary counters used by /api/debug_training_sessions."""
        events_query = """
        SELECT sessionID
        FROM events
        WHERE event IN ('ingressoPagina', 'uscitaPagina')
            AND pageName LIKE 'page-%'
            AND pageName <> 'page-survey.php'
        GROUP BY sessionID
        HAVING COUNT(DISTINCT pageName) = 10
        """

        quiz_distinct_page_query = """
        SELECT sessionID
        FROM quiz
        GROUP BY sessionID
        HAVING COUNT(DISTINCT pageName) = 10
        """

        quiz_count_query = """
        SELECT sessionID
        FROM quiz
        GROUP BY sessionID
        HAVING COUNT(*) >= 10
        """

        events_df = read_sql(events_query)
        quiz_distinct_df = read_sql(quiz_distinct_page_query)
        quiz_count_df = read_sql(quiz_count_query)

        events_sessions = set(events_df["sessionID"].dropna().astype(str))
        quiz_distinct_sessions = set(quiz_distinct_df["sessionID"].dropna().astype(str))
        quiz_count_sessions = set(quiz_count_df["sessionID"].dropna().astype(str))

        return {
                "complete_events_sessions": len(events_sessions),
                "quiz_sessions_distinct_pageName_10": len(quiz_distinct_sessions),
                "quiz_sessions_count_at_least_10": len(quiz_count_sessions),
                "intersection_events_and_quiz_pageName": len(events_sessions.intersection(quiz_distinct_sessions)),
                "intersection_events_and_quiz_count": len(events_sessions.intersection(quiz_count_sessions)),
        }
