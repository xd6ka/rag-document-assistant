"""Page-aware text chunking with overlap.

Chunks never cross a page boundary. That is a deliberate trade-off: slightly
worse retrieval context on pages that continue mid-sentence, but every chunk
carries an exact page number, which is what makes citations trustworthy.

Windows are ~``size`` characters, cut on whitespace/newline when possible so
words are not split in half, and consecutive windows overlap by ``overlap``
characters so a sentence straddling a boundary is still fully contained in at
least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import MIN_CHUNK_CHARS

_SEPARATORS = ("\n", " ", ".", ",")


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int   # 1-based page number in the source document
    index: int  # position of this chunk inside the page


def _windows(text: str, size: int, overlap: int):
    """Yield ``(start_offset, piece)`` windows of at most ``size`` characters."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    length = len(text)
    start = 0
    while start < length:
        end = min(start + size, length)

        if end < length:
            # Prefer a natural boundary in the second half of the window.
            for sep in _SEPARATORS:
                cut = text.rfind(sep, start + size // 2, end)
                if cut > start:
                    end = cut + 1 if sep == "." else cut
                    break

        piece = text[start:end].strip()
        if piece:
            yield start, piece

        if end >= length:
            break
        start = max(end - overlap, start + 1)


def chunk_page(text: str, page: int, size: int, overlap: int) -> list[Chunk]:
    """Split one page into overlapping chunks, merging a tiny trailing fragment."""
    pieces = [piece for _, piece in _windows(text or "", size, overlap)]

    if len(pieces) > 1 and len(pieces[-1]) < MIN_CHUNK_CHARS:
        pieces[-2] = f"{pieces[-2]} {pieces[-1]}".strip()
        pieces.pop()

    return [Chunk(text=piece, page=page, index=i) for i, piece in enumerate(pieces)]


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[Chunk]:
    """Chunk every page of a document, keeping exact page numbers."""
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        chunks.extend(chunk_page(page_text, page_number, size, overlap))
    return chunks
