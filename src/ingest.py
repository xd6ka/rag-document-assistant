"""End-to-end ingestion: PDF bytes -> page text -> chunks -> embeddings -> store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pypdf import PdfReader

from . import config
from .chunking import chunk_pages
from .embeddings import embed_texts
from .store import DocumentStore


def extract_pages(reader: PdfReader) -> list[str]:
    """Extract text page by page; a broken page yields "" instead of failing the job."""
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def ingest_pdf(
    source_name: str,
    pdf_file: Any,
    store: Optional[DocumentStore] = None,
    replace: bool = True,
) -> dict[str, Any]:
    """Index one PDF under ``source_name``.

    ``replace=True`` deletes the document's previous chunks first, so re-indexing
    an edited PDF never leaves stale chunks behind.
    """
    reader = PdfReader(pdf_file)
    pages = extract_pages(reader)
    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    if not chunks:
        return {"source": source_name, "pages": len(pages), "chunks": 0, "skipped": True}

    store = store or DocumentStore()
    if replace:
        store.delete_document(source_name)

    embeddings = embed_texts([chunk.text for chunk in chunks])
    written = store.add_document(source_name, chunks, embeddings)

    return {
        "source": source_name,
        "pages": len(pages),
        "chunks": written,
        "skipped": False,
    }


def ingest_path(path: str | Path, store: Optional[DocumentStore] = None) -> dict[str, Any]:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        return ingest_pdf(file_path.name, handle, store=store)
