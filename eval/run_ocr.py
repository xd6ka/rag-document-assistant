"""What does the OCR fallback actually buy? Measure it, do not claim it.

The corpus is rasterised page by page into image-only PDFs, which is exactly what
a scanner produces: same layout, same page count, no text layer. Then the same 25
in-scope questions run against three indexes:

    digital              the corpus as shipped (the ceiling, no OCR involved)
    scanned, no OCR      the rasterised corpus with the fallback switched off
    scanned, OCR         the rasterised corpus with the fallback on

Two things are reported: retrieval metrics per configuration, and how much of
each page's text the engine reproduced (word recall against the digital layer).
The recall number is what keeps the retrieval number honest — a suspiciously good
hit@1 on scanned pages would otherwise just mean the OCR invented the answer.

    python eval/run_ocr.py            # 300 dpi, ~5 s per page
    python eval/run_ocr.py --dpi 200  # cheaper, worse text

Everything is written to a throwaway store, so the working collection is untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from eval.run_eval import first_hit_rank, load_qa_set, read_page_texts  # noqa: E402
from src import config  # noqa: E402
from src.chunking import chunk_pages  # noqa: E402
from src.embeddings import embed_texts  # noqa: E402
from src.ocr import needs_ocr, ocr_page  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

RETRIEVE_K = 8


def word_counts(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def word_recall(truth: str, candidate: str) -> float:
    """Share of the page's words the OCR text reproduced (multiset, lowercased)."""
    expected = word_counts(truth)
    total = sum(expected.values())
    if not total:
        return 1.0
    got = word_counts(candidate)
    return sum(min(count, got.get(word, 0)) for word, count in expected.items()) / total


def rasterise(pdf: Path, dpi: int, target_dir: Path) -> Path:
    """Rewrite one PDF as page images only — the text layer disappears."""
    source = pymupdf.open(str(pdf))
    target_path = target_dir / pdf.name
    target = pymupdf.open()
    try:
        for page in source:
            png = page.get_pixmap(dpi=dpi).tobytes("png")
            blank = target.new_page(width=page.rect.width, height=page.rect.height)
            blank.insert_image(blank.rect, stream=png)
        target.save(str(target_path))
    finally:
        source.close()
        target.close()
    return target_path


def build_store(pages_by_doc: dict[str, list[str]], path: Path, ocr_source: dict[str, bytes] | None) -> dict:
    """Index the pages, optionally running OCR on thin ones.

    Returns the store, the post-OCR page text (so recall can be computed from
    exactly what was embedded) and per-page OCR timings.
    """
    store = DocumentStore(path=path, collection_name="ocr_eval")
    started = time.perf_counter()
    final_pages: dict[str, list[str]] = {}
    ocr_pages = 0
    ocr_timings: list[float] = []

    for source, pages in pages_by_doc.items():
        pages = list(pages)
        if ocr_source is not None:
            raw = ocr_source[source]
            for index, text in enumerate(pages):
                if not needs_ocr(text):
                    continue
                try:
                    page_started = time.perf_counter()
                    recognised = ocr_page(raw, index)
                    ocr_timings.append(time.perf_counter() - page_started)
                except Exception as error:  # pragma: no cover - engine-dependent
                    print(f"  ocr failed on {source} page {index + 1}: {error}")
                    continue
                if len(recognised.strip()) > len(text.strip()):
                    pages[index] = recognised
                    ocr_pages += 1

        final_pages[source] = pages
        chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if chunks:
            store.add_document(source, chunks, embed_texts([chunk.text for chunk in chunks]))

    return {
        "store": store,
        "pages": final_pages,
        "ocr_pages": ocr_pages,
        "ocr_seconds": ocr_timings,
        "seconds": time.perf_counter() - started,
    }


def score(store: DocumentStore, questions: list[dict]) -> dict:
    retriever = Retriever(store)
    ranks: list[int | None] = []
    started = time.perf_counter()
    for item in questions:
        contexts = retriever.retrieve(
            item["question"],
            k=config.CONTEXT_K,
            retrieve_k=RETRIEVE_K,
            rerank=True,
            cross_encoder=True,
        )
        ranks.append(first_hit_rank(contexts, f"{item['doc']}.pdf", item["gold_pages"]))
    elapsed = time.perf_counter() - started

    hits = [rank for rank in ranks if rank is not None]
    return {
        "chunks": store.count(),
        "hit_at_k": round(len(hits) / len(ranks), 3),
        "hit_at_1": round(sum(1 for rank in ranks if rank == 1) / len(ranks), 3),
        "mrr": round(sum(1.0 / rank for rank in hits) / len(ranks), 3),
        "seconds_per_question": round(elapsed / len(ranks), 3),
        "misses": [item["id"] for item, rank in zip(questions, ranks) if rank is None],
    }


def report_line(name: str, result: dict) -> str:
    return (
        f"{name:<18} hit@4={result['hit_at_k']} hit@1={result['hit_at_1']} "
        f"mrr={result['mrr']} chunks={result['chunks']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=config.OCR_DPI, help="rasterisation resolution")
    args = parser.parse_args()

    questions = [item for item in load_qa_set(ROOT / "eval" / "qa_set.jsonl") if item.get("in_scope", True)]
    digital_pages = read_page_texts(ROOT / "eval" / "corpus")

    workdir = Path(tempfile.mkdtemp(prefix="rag_ocr_"))
    scanned_dir = workdir / "scanned"
    scanned_dir.mkdir()

    print(f"rasterising {len(digital_pages)} documents at {args.dpi} dpi ...")
    started = time.perf_counter()
    scanned_bytes: dict[str, bytes] = {}
    for source in digital_pages:
        path = rasterise(ROOT / "eval" / "corpus" / source, args.dpi, scanned_dir)
        scanned_bytes[source] = path.read_bytes()
    raster_seconds = time.perf_counter() - started

    # Sanity check: the rasterised copies really have no text layer. If they did,
    # "scanned" would silently mean "digital" and the comparison would be fake.
    scanned_pages = read_page_texts(scanned_dir)
    leaked = sum(1 for pages in scanned_pages.values() for page in pages if page.strip())
    if leaked:
        raise SystemExit(f"rasterised copies still expose text on {leaked} pages")
    print(f"  {sum(len(p) for p in scanned_pages.values())} image-only pages, {raster_seconds:.1f}s to render")

    results: dict[str, dict] = {}

    digital = build_store(digital_pages, workdir / "s_digital", None)
    results["digital"] = {**score(digital["store"], questions), "index_seconds": round(digital["seconds"], 1), "ocr_pages": 0}
    print(report_line("digital", results["digital"]))

    empty = build_store(scanned_pages, workdir / "s_scanned", None)
    results["scanned_no_ocr"] = {**score(empty["store"], questions), "index_seconds": round(empty["seconds"], 1), "ocr_pages": 0}
    print(report_line("scanned, no OCR", results["scanned_no_ocr"]))

    print("running OCR on the scanned pages (the slow part) ...")
    ocr_run = build_store(scanned_pages, workdir / "s_ocr", scanned_bytes)
    results["scanned_ocr"] = {
        **score(ocr_run["store"], questions),
        "index_seconds": round(ocr_run["seconds"], 1),
        "ocr_pages": ocr_run["ocr_pages"],
    }
    print(report_line("scanned, OCR", results["scanned_ocr"]))

    # Page-level fidelity, computed from the exact text that was embedded.
    per_page: list[dict] = []
    for source, pages in digital_pages.items():
        for index, truth in enumerate(pages):
            if not truth.strip():
                continue
            per_page.append(
                {
                    "doc": source,
                    "page": index + 1,
                    "word_recall": round(word_recall(truth, ocr_run["pages"][source][index]), 3),
                }
            )
    recalls = [entry["word_recall"] for entry in per_page]
    mean_ocr = sum(ocr_run["ocr_seconds"]) / len(ocr_run["ocr_seconds"]) if ocr_run["ocr_seconds"] else 0.0
    print(f"word recall        mean={sum(recalls) / len(recalls):.3f} min={min(recalls):.3f} over {len(recalls)} pages, {mean_ocr:.1f}s/page")

    out_path = ROOT / "eval" / "reports" / "ocr.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "dpi": args.dpi,
                "n_questions": len(questions),
                "raster_seconds": round(raster_seconds, 1),
                "ocr_seconds_per_page": round(mean_ocr, 2),
                "word_recall": {
                    "mean": round(sum(recalls) / len(recalls), 3),
                    "min": round(min(recalls), 3),
                    "pages": per_page,
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
