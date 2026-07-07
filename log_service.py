from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from flask import url_for
from graphviz import Source
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.dfg import visualizer as dfg_visualizer


LOGGER = logging.getLogger("app")


def calculate_cyclomatic_complexity(df: pd.DataFrame) -> int:
    """Calculate cyclomatic complexity from the current page-level trace."""
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


def style_start_end_nodes(dot_source: str) -> str:
    """Apply layout refinements and style Start/End nodes in the rendered DFG."""
    styled_lines = []
    for line in dot_source.splitlines():
        if "->" in line:
            styled_lines.append(line)
            continue

        # Make the graph more linear and readable (left-to-right with extra spacing).
        if line.strip().startswith("graph ["):
            if "rankdir=" not in line:
                line = re.sub(r"\]$", ', rankdir=LR]', line)
            if "nodesep=" not in line:
                line = re.sub(r"\]$", ', nodesep="1.0"]', line)
            if "ranksep=" not in line:
                line = re.sub(r"\]$", ', ranksep="1.2"]', line)

        # PM4Py frequency labels can render as Start (1) / End (1): strip the count.
        line = re.sub(r'label="(Start|End) \(\d+\)"', r'label="\1"', line)

        if "label=\"Start\"" in line or "label=\"End\"" in line:
            if "shape=" not in line:
                line = re.sub(r"\]$", ', shape="circle", style="filled", fillcolor="white"]', line)
        styled_lines.append(line)

    return "\n".join(styled_lines)


def safe_session_id(session_id: str) -> str:
    return "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))


def export_session_event_log_csv(df: pd.DataFrame, session_id: str, logs_dir: Path) -> Path:
    """Write the current session event log to logs/<session_id>.csv and return the file path."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / f"{safe_session_id(session_id)}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def create_dfg_png(df: pd.DataFrame, session_id: str, dfg_dir: Path) -> str | None:
    """Create the current DFG with PM4Py, save it as PNG, and return the static file URL."""
    if df.empty or len(df) < 2:
        return None

    # Use page-entry events for DFG transitions so entry/exit pairs do not create artificial self-loops.
    # Real self-loops are still preserved when the same page is entered again consecutively.
    dfg_df = df.copy()
    if "event" in dfg_df.columns:
        dfg_df = dfg_df[dfg_df["event"] == "ingressoPagina"].copy()

    if dfg_df.empty:
        return None

    # Show real activities in uppercase for the rendered DFG only.
    dfg_df["concept:name"] = dfg_df["concept:name"].astype(str).str.upper()

    start_row = {
        "case:concept:name": session_id,
        "concept:name": "Start",
        "time:timestamp": dfg_df["time:timestamp"].min() - pd.Timedelta(seconds=1),
    }

    end_row = {
        "case:concept:name": session_id,
        "concept:name": "End",
        "time:timestamp": dfg_df["time:timestamp"].max() + pd.Timedelta(seconds=1),
    }

    event_log_df = pd.concat(
        [
            pd.DataFrame([start_row]),
            dfg_df[["case:concept:name", "concept:name", "time:timestamp"]],
            pd.DataFrame([end_row]),
        ],
        ignore_index=True,
    )

    parameters = {
        log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
    }

    event_log = log_converter.apply(
        event_log_df,
        parameters=parameters,
        variant=log_converter.Variants.TO_EVENT_LOG,
    )

    dfg = dfg_discovery.apply(event_log)
    gviz = dfg_visualizer.apply(dfg, log=event_log, variant=dfg_visualizer.Variants.FREQUENCY)

    safe_id = safe_session_id(session_id)
    png_path = dfg_dir / f"dfg_{safe_id}.png"

    styled_source = style_start_end_nodes(gviz.source)
    try:
        Source(styled_source, format="png").render(filename=str(png_path.with_suffix("")), cleanup=True)
        return url_for("serve_dfg", filename=png_path.name)
    except Exception:
        LOGGER.exception("DFG rendering failed for session %s", session_id)
        return None
