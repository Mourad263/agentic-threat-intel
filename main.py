"""Entry point for testing the full LangGraph pipeline."""

from __future__ import annotations

import re
from typing import Any

from dotenv import load_dotenv

from app.graph import build_graph
from app.state import create_initial_state
from app.utils.app_mode import is_demo_mode
from app.utils.execution_trace import summarize_documents, summarize_web_results

SECTION_PATTERN = re.compile(
    r"(?ms)^(Overview|Attack Explanation|Recent Examples|IOCs|Detection|Mitigation|Limitations)\s*:?\s*$"
)


def _extract_final_answer_sections(final_answer: str) -> dict[str, str]:
    """Parse the standard answer sections from the final answer text."""
    matches = list(SECTION_PATTERN.finditer(final_answer or ""))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        section_name = match.group(1)
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(final_answer)
        sections[section_name] = final_answer[content_start:content_end].strip()
    return sections


def _print_block(title: str, value: Any) -> None:
    """Print one labeled output block."""
    print(f"\n{title}:")
    if isinstance(value, (list, dict)):
        print(value)
        return
    print(value if value not in (None, "") else "None")


def print_verbose_pipeline_output(final_state: dict[str, Any]) -> None:
    """Print the full demo-style pipeline output."""
    retrieved_docs = final_state.get("retrieved_docs", []) or []
    web_results = final_state.get("web_results", []) or []

    print("\n" + "=" * 60)
    print("FINAL PIPELINE OUTPUT")
    print("=" * 60)

    _print_block("Topic", final_state.get("topic", ""))
    _print_block("Plan", final_state.get("plan", ""))
    _print_block("Retrieved docs count", len(retrieved_docs))
    _print_block("Short retrieved preview", summarize_documents(retrieved_docs))
    _print_block("Web results count", len(web_results))
    _print_block("Sample web results", summarize_web_results(web_results))
    _print_block("Critic Feedback", final_state.get("critic_feedback", ""))
    _print_block("Final Revised Answer", final_state.get("final_answer", ""))
    _print_block("Validation Report", final_state.get("validation_report", {}))
    _print_block("Final state keys", list(final_state.keys()))


def print_clean_output(final_state: dict[str, Any]) -> None:
    """Print only final answer and validation report."""
    final_answer = final_state.get("final_answer", "") or ""
    validation_report = final_state.get("validation_report", {}) or {}

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer if final_answer.strip() else "None")

    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(validation_report)


def main() -> None:
    """Ask for a query and run the full graph pipeline."""
    load_dotenv()

    try:
        user_query = input("Enter your query: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("No query provided.")
        return

    if not user_query:
        print("No query provided.")
        return

    state = create_initial_state(user_query)

    graph = build_graph()
    final_state = graph.invoke(state)

    if is_demo_mode():
        print_verbose_pipeline_output(final_state)
    else:
        print_clean_output(final_state)


if __name__ == "__main__":
    main()
