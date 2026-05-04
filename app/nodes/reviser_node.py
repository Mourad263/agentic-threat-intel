"""Reviser node for improving the final cybersecurity answer."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.state import AppState
from app.utils.prompts import load_prompt
from app.utils.debug import debug_print
from app.utils.execution_trace import record_execution_step, truncate_text
from app.nodes.writer_node import synthesize_recent_examples


def get_reviser_llm() -> ChatOllama:
    return ChatOllama(model="llama3.2", temperature=0.2)


def _clean_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _feedback_mentions_real_issue(critic_feedback: str) -> bool:
    feedback = critic_feedback.lower().strip()

    if not feedback:
        return False

    issue_markers = [
        "weak",
        "missing",
        "generic",
        "too generic",
        "snippet",
        "rewrite",
        "unclear",
        "unsupported",
        "needs",
        "avoid",
    ]

    return any(marker in feedback for marker in issue_markers)


def _recent_examples_feedback_is_weak(critic_feedback: str) -> bool:
    feedback = critic_feedback.lower()

    if "recent examples" not in feedback:
        return False

    weak_markers = [
        "weak",
        "generic",
        "snippet",
        "rewrite",
        "article-title",
        "title dumps",
        "statistics-only",
        "vague",
        "avoid",
    ]

    return any(marker in feedback for marker in weak_markers)


def _replace_recent_examples_section(draft_answer: str, new_examples: str) -> str:
    pattern = re.compile(
        r"(?mis)(^Recent Examples\s*:?\s*)(.*?)(?=^IOCs\s*:?\s*$|^Detection\s*:?\s*$|^Mitigation\s*:?\s*$|^Limitations\s*:?\s*$|\Z)"
    )

    if pattern.search(draft_answer):
        return pattern.sub(
            lambda match: f"{match.group(1)}{new_examples.strip()}\n\n",
            draft_answer,
            count=1,
        )

    return draft_answer


def _recent_examples_section_is_weak(draft_answer: str, web_results: list[dict]) -> bool:
    match = re.search(
        r"(?mis)^Recent Examples\s*:?\s*(.*?)(?=^IOCs\s*:?\s*$|^Detection\s*:?\s*$|^Mitigation\s*:?\s*$|^Limitations\s*:?\s*$|\Z)",
        draft_answer or "",
    )

    if not match:
        return True

    section = match.group(1).strip()
    normalized = re.sub(r"\s+", " ", section).strip().lower()

    if web_results and len(normalized.split()) < 60:
        return True

    weak_markers = [
        "responds to many",
        "statistics",
        "percentage",
        "trend",
        "didn't detect",
        "did not detect",
        "what is",
        "examples of",
        "guide",
        "glossary",
        "definition",
    ]

    if any(marker in normalized for marker in weak_markers):
        return True

    bullet_count = len(re.findall(r"(?m)^\s*[-*]\s+", section))

    if web_results and bullet_count < 2:
        return True

    analyst_markers = [
        "which shows",
        "which suggests",
        "this matters",
        "highlighting",
        "reinforcing",
        "indicating",
        "demonstrating",
    ]

    if web_results and not any(marker in normalized for marker in analyst_markers):
        return True

    return False


def reviser_node(state: AppState) -> AppState:
    draft_answer = state.get("draft_answer", "").strip()
    critic_feedback = state.get("critic_feedback", "").strip()
    web_results = state.get("web_results", [])

    debug_print("[reviser_node] started")
    debug_print(f"[reviser_node] draft length: {len(draft_answer)}")
    debug_print(f"[reviser_node] critic feedback length: {len(critic_feedback)}")

    recent_examples_need_fix = (
        _recent_examples_feedback_is_weak(critic_feedback)
        or _recent_examples_section_is_weak(draft_answer, web_results)
    )

    has_real_feedback = _feedback_mentions_real_issue(critic_feedback)

    if recent_examples_need_fix and web_results:
        improved_examples = synthesize_recent_examples(web_results)
        draft_answer = _replace_recent_examples_section(draft_answer, improved_examples)

        debug_print("[reviser_node] deterministically repaired Recent Examples section")

        state["draft_answer"] = _clean_output(draft_answer)

        record_execution_step(
            state,
            "reviser",
            title="Reviser",
            summary="Repaired weak Recent Examples using grounded web evidence without an extra LLM rewrite.",
            details={
                "skipped_llm": True,
                "recent_examples_repaired": True,
                "draft_preview": truncate_text(state["draft_answer"], 800),
            },
        )

        return state

    if not has_real_feedback:
        debug_print("[reviser_node] fast skip because no real critic feedback was found")

        state["draft_answer"] = draft_answer

        record_execution_step(
            state,
            "reviser",
            title="Reviser",
            summary="Skipped reviser because no meaningful critic feedback was found.",
            details={
                "skipped": True,
                "draft_preview": truncate_text(draft_answer, 800),
            },
        )

        return state

    system_prompt = load_prompt(
        "reviser_prompt.txt",
        """
You are a cybersecurity threat-intelligence reviser.

Improve only the weak parts identified by the critic.
Preserve the exact section structure:
Overview
Attack Explanation
Recent Examples
IOCs
Detection
Mitigation
Limitations

Rules:
- Do not invent incidents, victims, dates, actors, or technical details.
- Keep the answer concise.
- Strengthen weak SOC-style reasoning.
- Recent Examples must explain what happened and why it matters.
        """,
    )

    human_prompt = f"""
Draft:
{draft_answer[:1800]}

Critic Feedback:
{critic_feedback[:600]}

Revise the draft according to the feedback.
Return the full improved answer with the same section structure.
""".strip()

    try:
        response = get_reviser_llm().invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        revised_answer = response.content if isinstance(response.content, str) else str(response.content)

    except Exception:
        revised_answer = draft_answer

    cleaned_answer = _clean_output(revised_answer or draft_answer)

    state["draft_answer"] = cleaned_answer

    record_execution_step(
        state,
        "reviser",
        title="Reviser",
        summary="Applied LLM revision based on critic feedback.",
        details={
            "draft_preview": truncate_text(cleaned_answer, 800),
        },
    )

    return state