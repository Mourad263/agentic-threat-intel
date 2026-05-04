"""Planner node for the agentic threat intelligence workflow.

Fast version:
- No LLM call.
- Rule-based topic detection.
- Deterministic plan generation.
- Keeps the same state fields and execution trace structure.
"""

from __future__ import annotations

import re

from app.state import AppState
from app.utils.debug import debug_print
from app.utils.execution_trace import record_execution_step

TOPIC_KEYWORDS = {
    "Ransomware": [
        "ransomware",
        "locker",
        "crypto malware",
        "crypto-malware",
        "extortion malware",
    ],
    "DDos": [
        "ddos",
        "dos",
        "denial of service",
        "distributed denial of service",
        "botnet",
    ],
    "brute force": [
        "brute force",
        "password spray",
        "password spraying",
        "credential stuffing",
        "credential attack",
    ],
    "trojan": [
        "trojan",
        "backdoor",
        "remote access trojan",
        "rat",
        "malware backdoor",
    ],
}

REQUIRED_SECTIONS = [
    "Overview",
    "Attack Explanation",
    "Recent Examples",
    "IOCs",
    "Detection",
    "Mitigation",
    "Limitations",
]

RECENT_INTENT_TERMS = {
    "recent",
    "latest",
    "current",
    "today",
    "this week",
    "news",
    "examples",
    "incidents",
    "campaigns",
    "active",
    "ongoing",
}

WEB_INTENT_TERMS = {
    "recent",
    "latest",
    "current",
    "today",
    "this week",
    "news",
    "cve",
    "breach",
    "campaign",
    "campaigns",
    "active",
    "ongoing",
    "examples",
    "incidents",
}

EXPLANATION_INTENT_TERMS = {
    "what is",
    "explain",
    "how does",
    "overview",
    "definition",
    "concept",
    "difference",
}


def detect_topic(user_query: str) -> str:
    """Detect which topic folder best matches the user query."""
    query = user_query.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", query):
                return topic

    return "Ransomware"


def _detect_route_hint(user_query: str) -> str:
    """Infer whether the query likely needs RAG, web, or both.

    The real router still makes the final route decision in graph.py.
    This is only a readable planner hint.
    """
    query = user_query.lower()

    wants_web = any(term in query for term in WEB_INTENT_TERMS)
    wants_explanation = any(term in query for term in EXPLANATION_INTENT_TERMS)

    if wants_web and wants_explanation:
        return "both"

    if wants_web:
        return "web"

    return "rag"


def _query_requests_recent_examples(user_query: str) -> bool:
    """Return True when the user asks for current examples/incidents."""
    query = user_query.lower()
    return any(term in query for term in RECENT_INTENT_TERMS)


def _build_plan(user_query: str, topic: str) -> str:
    """Build a deterministic plan without calling the LLM."""
    route_hint = _detect_route_hint(user_query)
    recent_needed = _query_requests_recent_examples(user_query)

    sections = "\n".join(f"- {section}" for section in REQUIRED_SECTIONS)

    recent_instruction = (
        "Recent examples required: Yes — use live web results and avoid old or generic examples."
        if recent_needed
        else "Recent examples required: No explicit request, but include the section if evidence is available."
    )

    return (
        f"Topic: {topic}\n"
        f"Required sections:\n"
        f"{sections}\n"
        f"Retrieval needed: Yes\n"
        f"Route hint: {route_hint}\n"
        f"{recent_instruction}"
    )


def planner_node(state: AppState) -> AppState:
    """Generate the plan and selected topic using deterministic logic."""
    user_query = state["user_query"]
    topic = detect_topic(user_query)
    plan = _build_plan(user_query=user_query, topic=topic)

    debug_print(f"\nDetected topic folder: {topic}")
    debug_print("\nPlan:")
    debug_print(plan)

    state["topic"] = topic
    state["plan"] = plan

    record_execution_step(
        state,
        "planner",
        title="Planner",
        summary=f"Selected topic `{topic}` and generated a deterministic retrieval plan.",
        details={
            "topic": topic,
            "plan": plan,
            "llm_used": False,
        },
    )

    return state


def run_planner(state: AppState) -> AppState:
    """Backward-compatible wrapper that mutates and returns full state."""
    state.update(planner_node(state))
    return state