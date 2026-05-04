"""Shared state definition for the minimal planner flow."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document


class AppState(TypedDict):
    """Simple state passed between functions."""

    user_query: str
    topic: str
    plan: str
    route: str
    retrieved_docs: list[Document]
    web_results: list
    draft_answer: str
    critic_feedback: str
    final_answer: str
    validation_report: dict[str, object]
    execution_trace: dict[str, dict[str, Any]]
    fast_path_used: bool


def create_initial_state(user_query: str) -> AppState:
    """Create the starting state for the app."""
    return {
        "user_query": user_query,
        "topic": "",
        "plan": "",
        "route": "",
        "retrieved_docs": [],
        "web_results": [],
        "draft_answer": "",
        "critic_feedback": "",
        "final_answer": "",
        "validation_report": {},
        "execution_trace": {},
        "fast_path_used": False,
    }