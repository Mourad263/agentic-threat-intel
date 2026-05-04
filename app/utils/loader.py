"""Utilities for loading local PDF files."""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from app.utils.debug import debug_print


def load_documents(data_dir: str = "data/raw_docs", topic: str | None = None) -> list[Document]:
    """Load PDF files from all topics or from one selected topic folder."""
    base_path = Path(data_dir)

    if topic:
        base_path = base_path / topic

    pdf_paths = sorted(base_path.rglob("*.pdf"))

    debug_print(f"\nSearching for PDFs in: {base_path.resolve()}")
    if topic:
        debug_print(f"Loading topic folder only: {topic}")

    if not pdf_paths:
        debug_print("No PDF files were found.")
        return []

    all_documents: list[Document] = []

    # Load each PDF one by one so it is easy to see what is happening.
    for pdf_path in pdf_paths:
        debug_print(f"Loading: {pdf_path}")

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            all_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_number,
                        "topic": topic or pdf_path.parent.name,
                    },
                )
            )

    debug_print(f"Loaded {len(all_documents)} document pages from {len(pdf_paths)} PDF files.")

    all_documents = all_documents[:100]

    debug_print(f"Using only {len(all_documents)} pages for faster processing")

    return all_documents
