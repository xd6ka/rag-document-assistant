"""Render the evaluation corpus as PDFs.

    python eval/build_corpus.py

Writes one PDF per document into eval/corpus/, one PDF page per page in the
content definition. Deterministic output: same content, same page count, same
page boundaries, which is what makes the gold page labels in eval/qa_set.jsonl
valid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.corpus_source import DOCS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "corpus"

TITLE_STYLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=12)
BODY_STYLE = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=10)


def build_document(doc_id: str, spec: dict, out_dir: Path) -> Path:
    path = out_dir / f"{doc_id}.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=spec["title"],
        author="eval corpus",
    )

    story = []
    for page_index, paragraphs in enumerate(spec["pages"], start=1):
        if page_index == 1:
            story.append(Paragraph(escape(spec["title"]), TITLE_STYLE))
        else:
            story.append(Paragraph(escape(f"{spec['title']} - page {page_index}"), TITLE_STYLE))
        for text in paragraphs:
            story.append(Paragraph(escape(text), BODY_STYLE))
        story.append(Spacer(1, 6))
        if page_index < len(spec["pages"]):
            story.append(PageBreak())

    doc.build(story)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for doc_id, spec in DOCS.items():
        path = build_document(doc_id, spec, OUT_DIR)
        manifest[doc_id] = {"title": spec["title"], "pages": len(spec["pages"]), "bytes": path.stat().st_size}
        print(f"{path.name}: {len(spec['pages'])} pages, {path.stat().st_size} bytes")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"corpus written to {OUT_DIR}")


if __name__ == "__main__":
    main()
