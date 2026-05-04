"""Final validator node for lightweight output safety and quality checks."""

from __future__ import annotations

from app.state import AppState
from app.utils.debug import debug_print
from app.utils.execution_trace import record_execution_step
from app.utils.output_validator import validate_output


def validator_node(state: AppState) -> AppState:
    """Apply safe post-processing to the final answer without regenerating it."""
    candidate_answer = state.get("draft_answer", "").strip()
    web_results = state.get("web_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    topic = state.get("topic", "")

    debug_print("[validator_node] started")
    debug_print(f"[validator_node] candidate answer length: {len(candidate_answer)}")
    debug_print(f"[validator_node] topic: {topic}")
    debug_print(f"[validator_node] web results count: {len(web_results)}")
    debug_print(f"[validator_node] retrieved docs count: {len(retrieved_docs)}")

    final_answer, validation_report = validate_output(
        candidate_answer,
        web_results=web_results,
        retrieved_docs=retrieved_docs,
        topic=topic,
    )

    state["final_answer"] = final_answer
    state["validation_report"] = validation_report
    state["draft_answer"] = final_answer

    record_execution_step(
        state,
        "validator",
        title="Validator",
        summary="Validated and finalized the answer without changing the production output schema.",
        details={
            "topic": topic,
            "final_answer": final_answer,
            "validation_report": validation_report,
        },
    )

    return state