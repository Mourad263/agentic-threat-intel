"""Small helper for creating the local Ollama chat model."""

from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    """Return the local Ollama chat model."""
    return ChatOllama(model="llama3.2", temperature=0.3)
