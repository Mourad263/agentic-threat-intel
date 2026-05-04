"""Simple helpers for storing and searching documents with Chroma."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.debug import debug_print
from app.utils.loader import load_documents

# Save the Chroma database locally on disk.
VECTORSTORE_DIR = Path.home() / ".codex" / "memories" / "agentic-threat-intel-chroma"
EMBEDDING_MODEL = "all-minilm"


def get_embeddings():
    """Create a real Ollama embedding model for RAG."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def _collection_name_for_topic(topic: str) -> str:
    """Create a simple Chroma collection name for one topic."""
    normalized_topic = topic.strip().lower().replace(" ", "_")
    return f"threat_intel_{normalized_topic}"


def get_vectorstore(topic: str) -> Chroma:
    """Create or load the local Chroma vector store for one topic."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=_collection_name_for_topic(topic),
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=get_embeddings(),
    )


def _split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks before storing them."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)


def ingest_documents(topic: str, documents: list[Document] | None = None) -> Chroma:
    """Load documents for one topic into its local Chroma collection."""
    vectorstore = get_vectorstore(topic)
    current_count = vectorstore._collection.count()

    # Avoid adding the same topic chunks again if the collection already has data.
    if current_count > 0:
        debug_print(f"\nUsing existing vector store for {topic} with {current_count} chunks.")
        return vectorstore

    if documents is None:
        documents = load_documents(topic=topic)

    if not documents:
        debug_print("\nNo documents were loaded, so nothing was added to Chroma.")
        return vectorstore

    chunks = _split_documents(documents)
    debug_print(f"\nCreating vector store for {topic} with {len(chunks)} chunks...")
    vectorstore.add_documents(chunks)
    debug_print(f"Finished ingesting {topic} documents into Chroma.")

    return vectorstore


def retrieve_relevant_chunks(query: str, topic: str, top_k: int = 3) -> list[Document]:
    """Return the most relevant document chunks for a query within one topic."""
    vectorstore = ingest_documents(topic=topic)
    return vectorstore.similarity_search(query, k=top_k)
