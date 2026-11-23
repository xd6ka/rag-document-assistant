"""OCR fallback tests: the switch, the replacement rule and the failure path.

The engine itself is faked here. What matters is not that PP-OCR reads a glyph
well (it does, that is measured in `eval/run_ocr.py`), but that a scanned page is
sent to OCR, a normal page never is, and a broken engine cannot take down an
upload.
"""

from __future__ import annotations

import pymupdf
import pytest

from src import config, ingest, ocr
from src.ocr import needs_ocr


class FakeStore:
    """Records what ingest writes, so the tests do not touch ChromaDB."""

    def __init__(self):
        self.added: dict[str, list[str]] = {}
        self.deleted: list[str] = []

    def delete_document(self, source):
        self.deleted.append(source)
        return 1

    def add_document(self, source, chunks, embeddings):
        self.added[source] = [chunk.text for chunk in chunks]
        return len(chunks)


def text_pdf(text: str) -> bytes:
    """One-page PDF with a real text layer."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def scanned_pdf(text: str) -> bytes:
    """One-page PDF that holds an image instead of text — what a scanner produces."""
    source = pymupdf.open()
    page = source.new_page()
    page.insert_text((72, 72), text, fontsize=14)
    png = page.get_pixmap(dpi=150).tobytes("png")
    source.close()

    document = pymupdf.open()
    image_page = document.new_page()
    image_page.insert_image(image_page.rect, stream=png)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture(autouse=True)
def no_embeddings(monkeypatch):
    # Indexing would otherwise load the real embedder (~5 s) on every case.
    monkeypatch.setattr(ingest, "embed_texts", lambda texts, **kwargs: [[0.0] * 8 for _ in texts])
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(config, "CHUNK_SIZE", 800)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 150)


def test_needs_ocr_only_for_thin_text():
    assert needs_ocr("") is True
    assert needs_ocr("Page 3") is True
    assert needs_ocr("A real paragraph " * 20) is False


def test_scanned_page_is_ocr_ed(monkeypatch, store):
    recognised = "OCR TEXT " * 20
    calls: list[int] = []

    def fake_ocr(pdf_bytes, page_index, dpi=None):
        calls.append(page_index)
        return recognised

    monkeypatch.setattr(ocr, "ocr_page", fake_ocr)

    result = ingest.ingest_pdf("scan.pdf", scanned_pdf("Scanned page body"), store=store)

    assert calls == [0]
    assert result["ocr_pages"] == 1
    assert recognised.strip() in " ".join(store.added["scan.pdf"])


def test_page_with_text_layer_never_reaches_ocr(monkeypatch, store):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("OCR ran on a page that already had text")

    monkeypatch.setattr(ocr, "ocr_page", explode)

    result = ingest.ingest_pdf("normal.pdf", text_pdf("Plain text layer, plenty of it. " * 10), store=store)

    assert result["ocr_pages"] == 0
    assert result["chunks"] > 0


def test_ocr_failure_keeps_the_document_and_reports_a_scan(monkeypatch, store):
    def broken(pdf_bytes, page_index, dpi=None):
        raise RuntimeError("onnxruntime session failed")

    monkeypatch.setattr(ocr, "ocr_page", broken)

    result = ingest.ingest_pdf("broken.pdf", scanned_pdf("Scanned page body"), store=store)

    assert result["ocr_pages"] == 0
    assert result["chunks"] == 0
    # Empty pages with no OCR text are reported as a scan, not as "0 chunks" silence.
    assert result["scanned"] is True


def test_ocr_disabled_skips_the_engine(monkeypatch, store):
    monkeypatch.setattr(config, "OCR_ENABLED", False)

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("OCR ran while disabled")

    monkeypatch.setattr(ocr, "ocr_page", explode)

    result = ingest.ingest_pdf("scan.pdf", scanned_pdf("Scanned page body"), store=store)

    assert result["ocr_pages"] == 0
    assert result["scanned"] is True


def test_shorter_ocr_text_does_not_replace_the_text_layer(monkeypatch, store):
    """A noisy engine must not overwrite a page that already reads fine."""
    monkeypatch.setattr(config, "OCR_TRIGGER_CHARS", 200)
    monkeypatch.setattr(ocr, "ocr_page", lambda *a, **k: "ok")

    body = "This page has a real text layer. " * 4  # under the lowered trigger, over OCR's output
    result = ingest.ingest_pdf("mixed.pdf", text_pdf(body), store=store)

    assert result["ocr_pages"] == 0
    assert "real text layer" in " ".join(store.added["mixed.pdf"])
