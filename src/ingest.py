"""End-to-end ingestion: PDF bytes -> page text -> chunks -> embeddings -> store.

Two ways a page can produce text: the embedded text layer, or OCR when there is
no usable layer at all. OCR is attempted only for thin pages and its result is
kept only when it is clearly longer than what the text layer gave, so a normal
document is indexed exactly as before and a scanned page stops being a silent
empty string.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from pypdf import PdfReader

from . import config
from .chunking import chunk_pages
from .clean import strip_repeated_lines
from .embeddings import embed_texts
from .store import DocumentStore


def _pdf_bytes(pdf_file: Any) -> bytes:
    """Normalise a path, a file object or raw bytes into bytes."""
    if isinstance(pdf_file, (bytes, bytearray)):
        return bytes(pdf_file)
    if isinstance(pdf_file, (str, Path)):
        return Path(pdf_file).read_bytes()
    if hasattr(pdf_file, "getvalue"):
        return bytes(pdf_file.getvalue())
    pdf_file.seek(0)
    return bytes(pdf_file.read())


def extract_pages(reader: PdfReader) -> list[str]:
    """Extract text page by page; a broken page yields "" instead of failing the job."""
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def extract_pages_with_ocr(reader: PdfReader, pdf_bytes: bytes) -> tuple[list[str], int]:
    """Page text with an OCR fallback for pages that have no usable text layer.

    Returns the pages and how many of them needed OCR. An OCR failure is logged
    into the count only when it produced text — a crash is swallowed so that one
    unreadable page cannot abort an upload.
    """
    from .ocr import needs_ocr, ocr_page, ocr_available

    pages = extract_pages(reader)
    if not config.OCR_ENABLED:
        return pages, 0

    thin_pages = [index for index, text in enumerate(pages) if needs_ocr(text)]
    if not thin_pages or not ocr_available():
        return pages, 0

    used = 0
    for index in thin_pages:
        try:
            ocr_text = ocr_page(pdf_bytes, index)
        except Exception:
            continue  # keep the empty text layer rather than failing the document
        if len(ocr_text.strip()) > max(len(pages[index].strip()), config.OCR_TRIGGER_CHARS):
            pages[index] = ocr_text
            used += 1
    return pages, used


def strip_boilerplate(pages: list[str]) -> tuple[list[str], dict]:
    """Remove running headers/footers before chunking (config switchable)."""
    if not config.STRIP_BOILERPLATE:
        return pages, {"patterns": 0, "lines_removed": 0, "chars_removed": 0}
    return strip_repeated_lines(pages, min_page_share=config.BOILERPLATE_PAGE_SHARE)


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
    raw = _pdf_bytes(pdf_file)
    reader = PdfReader(BytesIO(raw))
    pages, ocr_pages = extract_pages_with_ocr(reader, raw)
    pages, boilerplate = strip_boilerplate(pages)
    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    if not chunks:
        return {
            "source": source_name,
            "pages": len(pages),
            "chunks": 0,
            "skipped": True,
            "ocr_pages": ocr_pages,
            "boilerplate": boilerplate,
            "scanned": bool(pages) and ocr_pages == 0,
        }

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
        "ocr_pages": ocr_pages,
        "boilerplate": boilerplate,
        # A document with pages but no chunks and no OCR text is almost certainly a
        # scan whose engine is missing; the UI can say so instead of showing "0 chunks".
        "scanned": bool(pages) and ocr_pages == 0 and not any(page.strip() for page in pages),
    }


def ingest_path(path: str | Path, store: Optional[DocumentStore] = None) -> dict[str, Any]:
    file_path = Path(path)
    return ingest_pdf(file_path.name, file_path, store=store)
