"""Helpers for production-safe pipeline output formatting."""

from __future__ import annotations

import json
from typing import Any

from app.utils.app_mode import is_demo_mode
from app.utils.execution_trace import format_trace_value

TRACE_STEP_ORDER = [
    "planner",
    "router",
    "retriever",
    "web_search",
    "writer",
    "critic",
    "reviser",
    "validator",
]


def build_pipeline_output(final_state: dict[str, Any]) -> dict[str, Any]:
    """Return the minimal production-facing output payload."""
    retrieved_docs = final_state.get("retrieved_docs", [])
    web_results = final_state.get("web_results", [])
    metadata = {
        "topic": final_state.get("topic", ""),
        "route": final_state.get("route", ""),
        "retrieved_docs_count": len(retrieved_docs),
        "web_results_count": len(web_results),
    }
    return {
        "final_answer": final_state.get("final_answer", ""),
        "validation_report": final_state.get("validation_report", {}),
        "metadata": metadata,
    }


def render_pipeline_output(final_state: dict[str, Any]) -> str:
    """Serialize the production-facing output payload."""
    return json.dumps(build_pipeline_output(final_state), indent=2, ensure_ascii=True)


def render_demo_output(final_state: dict[str, Any]) -> str:
    """Render a readable walkthrough of the pipeline for demo mode."""
    trace = final_state.get("execution_trace", {}) or {}
    sections: list[str] = []

    for step_name in TRACE_STEP_ORDER:
        step = trace.get(step_name)
        if not step:
            continue

        title = step.get("title", step_name.title())
        summary = step.get("summary", "")
        details = step.get("details", {}) or {}

        lines = [f"=== {title.upper()} ==="]
        if summary:
            lines.append(summary)
        for key, value in details.items():
            lines.append(f"{key}:")
            lines.append(format_trace_value(value))
        sections.append("\n".join(lines))

    sections.append("=== FINAL OUTPUT ===")
    sections.append(render_pipeline_output(final_state))
    return "\n\n".join(sections)


def render_app_output(final_state: dict[str, Any]) -> str:
    """Render the correct output format for the active app mode."""
    if is_demo_mode():
        return render_demo_output(final_state)
    return render_pipeline_output(final_state)
