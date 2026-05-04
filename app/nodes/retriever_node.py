"""Retriever node for the agentic threat intelligence workflow."""

from langchain_core.documents import Document

from app.state import AppState
from app.utils.debug import debug_print
from app.utils.execution_trace import record_execution_step, summarize_documents
from app.utils.vectorstore import retrieve_relevant_chunks


def _print_retrieved_previews(topic: str, retrieved_docs: list[Document]) -> None:
    """Print short previews of retrieved chunks for debugging."""
    debug_print(f"\nRetrieved document previews for topic: {topic}")

    for index, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "unknown")
        text = doc.page_content.strip().replace("\n", " ")
        preview = text[:150]

        debug_print(f"{index}. {source} | page {page}")
        debug_print(f"   {preview}...")


def retriever_node(state: AppState) -> AppState:
    """Retrieve relevant chunks and return the updated shared state."""
    user_query = state["user_query"]
    topic = state["topic"]
    retrieved_docs = retrieve_relevant_chunks(user_query, topic=topic, top_k=3)

    _print_retrieved_previews(topic, retrieved_docs)
    state["retrieved_docs"] = retrieved_docs
    record_execution_step(
        state,
        "retriever",
        title="Retriever",
        summary=f"Retrieved {len(retrieved_docs)} local document chunks for topic `{topic}`.",
        details={
            "topic": topic,
            "query": user_query,
            "retrieved_docs_count": len(retrieved_docs),
            "documents": summarize_documents(retrieved_docs),
        },
    )
    return state


def run_retriever(state: AppState) -> AppState:
    """Backward-compatible wrapper that mutates and returns full state."""
    state.update(retriever_node(state))
    return state
