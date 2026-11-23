"""OCR fallback for pages that carry no text layer.

A scanned PDF is a stack of images: ``pypdf`` returns an empty string for every
page, the corpus looks empty, and the only honest failure message is "no
extractable text". This module rasterises such a page with PyMuPDF and reads it
with RapidOCR (PP-OCR models on ONNX runtime).

Why this engine: it is pure ``pip`` — no ``tesseract.exe`` to install, no system
package, no model download at first use (the models ship inside the wheel), so the
offline promise the rest of the project makes survives. The cost is CPU time, and
that cost is measured rather than guessed: at 300 dpi one page takes ~5 s and
reproduces 99 % of the page's words on the benchmark corpus, against 84 % at 200
dpi and 62 % at 150 dpi. Hence the default resolution.

Everything here is lazy: importing this module must not import numpy, PyMuPDF or
onnxruntime, because the app imports it on every Streamlit rerun and most
documents never need OCR.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from . import config


class OcrUnavailable(RuntimeError):
    """Raised when the OCR engine cannot be imported on this machine."""


def ocr_available() -> bool:
    """True when the OCR engine can be imported (it can still fail at run time)."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def get_engine():
    """Lazy singleton: the ONNX session costs seconds and ~100 MB, load it once."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:  # pragma: no cover - depends on the machine
        raise OcrUnavailable(f"rapidocr is not importable: {exc}") from exc
    return RapidOCR()


def render_page_png(pdf_bytes: bytes, page_index: int, dpi: Optional[int] = None) -> bytes:
    """Rasterise one page of a PDF (given as bytes) into PNG."""
    import pymupdf  # local import: PyMuPDF is only needed when OCR actually runs

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = document[page_index]
        pixmap = page.get_pixmap(dpi=dpi or config.OCR_DPI)
        return pixmap.tobytes("png")
    finally:
        document.close()


def _png_to_bgr(png: bytes):
    """Decode PNG bytes into the BGR array RapidOCR expects."""
    import cv2
    import numpy as np

    array = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError("could not decode the rendered page")
    return array


def _join_lines(result, min_confidence: float) -> str:
    """Concatenate recognised lines, dropping boxes the detector was unsure about."""
    lines: list[str] = []
    for item in result or []:
        text = str(item[1])
        score = float(item[2]) if len(item) > 2 else 1.0
        if score >= min_confidence and text.strip():
            lines.append(text.strip())
    return "\n".join(lines)


def ocr_page(pdf_bytes: bytes, page_index: int, dpi: Optional[int] = None) -> str:
    """Read one page image into text. Raises ``OcrUnavailable`` without an engine."""
    engine = get_engine()
    png = render_page_png(pdf_bytes, page_index, dpi=dpi)
    result, _ = engine(_png_to_bgr(png))
    return _join_lines(result, config.OCR_MIN_CONFIDENCE)


def needs_ocr(text: str) -> bool:
    """True when the extracted text is too thin to be a real text layer.

    A page whose text layer exists is never sent through the engine, so indexing
    a normal PDF pays nothing for this feature.
    """
    return len("".join(text.split())) < config.OCR_TRIGGER_CHARS
