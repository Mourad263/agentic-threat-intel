"""Utilities for recording structured pipeline execution steps."""

from __future__ import annotations

import json
from typing import Any


def truncate_text(value: str, max_chars: int = 500) -> str:
    """Return a compact single-line preview of longer text."""
    normalized = " ".join((value or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def summarize_documents(documents: list[Any], limit: int = 3) -> list[dict[str, Any]]:
    """Convert retrieved documents into compact demo-friendly summaries."""
    summaries: list[dict[str, Any]] = []
    for doc in (documents or [])[:limit]:
        metadata = getattr(doc, "metadata", {}) or {}
        summaries.append(
            {
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "unknown"),
                "preview": truncate_text(getattr(doc, "page_content", ""), max_chars=180),
            }
        )
    return summaries


def summarize_web_results(results: list[dict[str, Any]], limit: int = 3) -> list[dict[str, str]]:
    """Convert web results into compact demo-friendly summaries."""
    summaries: list[dict[str, str]] = []
    for result in (results or [])[:limit]:
        summaries.append(
            {
                "title": result.get("title", "Untitled result"),
                "url": result.get("url", ""),
                "preview": truncate_text(result.get("content", ""), max_chars=180),
            }
        )
    return summaries


def record_execution_step(
    state: dict[str, Any],
    step: str,
    *,
    title: str,
    summary: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Upsert one structured execution-trace step into shared state."""
    trace = state.setdefault("execution_trace", {})
    trace[step] = {
        "title": title,
        "summary": summary,
        "details": details or {},
    }


def format_trace_value(value: Any) -> str:
    """Render trace values in a deterministic, readable format."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=True)
