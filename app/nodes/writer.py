"""Backward-compatible writer module."""

from app.nodes.writer_node import (
    build_web_evidence_block,
    format_retrieved_docs,
    format_retrieved_context,
    format_single_document,
    get_writer_llm,
    synthesize_recent_examples,
    writer_node,
)


def format_web_results(results, max_results=3):
    """Backward-compatible alias for compressed web evidence formatting."""
    return build_web_evidence_block(results[:max_results], max_results=max_results)
