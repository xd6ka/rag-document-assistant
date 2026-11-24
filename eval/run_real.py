"""Evaluate the pipeline on PDFs nobody wrote for this project.

The synthetic corpus in eval/corpus/ is friendly by construction: prose in two
columns, one idea per page, no running headers. Here the same retrieval and
answer code is pointed at six genuine documents (arXiv papers, a NIST
standard, a government report, a bill) with layouts that were never designed to
be chunked.

    python eval/fetch_real_corpus.py     # download once (not committed)
    python eval/run_real.py              # retrieval + answers
    python eval/run_real.py --no-answers # retrieval only, no model calls

Two things are measured that the synthetic set could not show:
  * page-level retrieval on real layouts (two columns, tables, numbered clauses)
  * how much repeated boilerplate (running headers) ends up inside the chunks
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import (  # noqa: E402
    CITATION_PATTERN,
    cited_blocks,
    is_refusal,
    load_qa_set,
    validate_qa_set,
)
from src import config  # noqa: E402
from src.citations import check_answer, support_rate  # noqa: E402
from src.chunking import chunk_pages  # noqa: E402
from src.generation import answer_question  # noqa: E402
from src.ingest import extract_pages, ingest_pdf  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

CORPUS_DIR = ROOT / "eval" / "real_corpus"
# The bill is 1039 pages and the report 585; indexing all of them would spend
# most of the CPU budget on pages no question asks about. Cap every long
# document at the same number of pages so the comparison stays symmetric.
PAGE_CAP = 60


def truncate(pdf_path: Path, max_pages: int) -> tuple[bytes, int, int]:
    """Return PDF bytes limited to the first ``max_pages`` pages, plus counts."""
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    if max_pages and total > max_pages:
        writer = PdfWriter()
        for page in reader.pages[:max_pages]:
            writer.add_page(page)
        buffer = tempfile.SpooledTemporaryFile()
        writer.write(buffer)
        buffer.seek(0)
        return buffer.read(), max_pages, total
    return pdf_path.read_bytes(), total, total


def page_texts(pdf_bytes: bytes) -> list[str]:
    return extract_pages(PdfReader(_bytes_io(pdf_bytes)))


def _bytes_io(raw: bytes):
    from io import BytesIO

    return BytesIO(raw)


def boilerplate(pages: list[str]) -> dict:
    """Find the line that repeats at the top of most pages (a running header).

    Every chunk keeps its page text verbatim, so a header costs characters on
    every page and gives the embedder a phrase that appears in every document
    section - it is a useful thing to quantify instead of merely suspecting.
    """
    heads = []
    for text in pages:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        heads.append(lines[0] if lines else "")
    if not heads:
        return {"header": None, "pages_with_header": 0, "page_share": 0.0, "char_share": 0.0}

    header, count = Counter(heads).most_common(1)[0]
    total_chars = sum(len(p or "") for p in pages) or 1
    if count <= 1 or not header:
        return {"header": None, "pages_with_header": 0, "page_share": 0.0, "char_share": 0.0}
    header_chars = sum(len(h) for h in heads if h == header)
    return {
        "header": header[:120],
        "pages_with_header": count,
        "page_share": round(count / len(pages), 3),
        "char_share": round(header_chars / total_chars, 4),
    }


def chunk_stats(pages: list[str]) -> dict:
    """Chunk count and size distribution produced by the real chunker."""
    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        return {"chunks": 0, "mean_chars": 0, "min_chars": 0, "max_chars": 0, "tiny": 0}
    sizes = [len(chunk.text) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "mean_chars": round(sum(sizes) / len(sizes)),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "tiny": sum(1 for size in sizes if size < config.MIN_CHUNK_CHARS),
    }


def first_hit_rank(contexts, gold_doc: str, gold_pages: list[int]) -> int | None:
    for rank, chunk in enumerate(contexts, start=1):
        if chunk.source == gold_doc and chunk.page in gold_pages:
            return rank
    return None


def rate(values) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(1 for value in usable if value) / len(usable), 3) if usable else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the real-world PDFs")
    parser.add_argument("--corpus", default=str(CORPUS_DIR))
    parser.add_argument("--qa", default=str(ROOT / "eval" / "qa_set_real.jsonl"))
    parser.add_argument("--max-pages", type=int, default=PAGE_CAP, help="pages per document (0 = all)")
    parser.add_argument("--k", type=int, default=config.CONTEXT_K)
    parser.add_argument("--retrieve-k", type=int, default=config.RETRIEVE_K)
    parser.add_argument("--no-answers", action="store_true")
    parser.add_argument("--no-ce", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "eval" / "reports" / "real_corpus.json"))
    args = parser.parse_args()

    corpus = Path(args.corpus)
    pdfs = sorted(corpus.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {corpus}. Run: python eval/fetch_real_corpus.py")

    items = load_qa_set(Path(args.qa))
    docs_by_name = {pdf.stem: pdf for pdf in pdfs}
    unknown = [item["doc"] for item in items if item.get("doc") and item["doc"] not in docs_by_name]
    if unknown:
        raise SystemExit(f"QA set names documents that are not in {corpus}: {sorted(set(unknown))}")

    # Gold pages are original page numbers, so only questions inside the cap can
    # be checked; anything deeper would point at a page that was never indexed.
    items = [item for item in items if all(page <= args.max_pages or args.max_pages == 0 for page in (item.get("gold_pages") or []))]
    if args.limit:
        items = items[: args.limit]

    workdir = Path(tempfile.mkdtemp(prefix="rag_real_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="real")

    doc_reports: list[dict] = []
    pages_by_doc: dict[str, list[str]] = {}
    truncated: dict[str, bytes] = {}

    for name, pdf in docs_by_name.items():
        raw, used_pages, total_pages = truncate(pdf, args.max_pages)
        truncated[name] = raw
        pages = page_texts(raw)
        pages_by_doc[f"{name}.pdf"] = pages

        started = time.perf_counter()
        result = ingest_pdf(f"{name}.pdf", raw, store=store)
        seconds = round(time.perf_counter() - started, 1)

        stats = chunk_stats(pages)
        info = {
            "document": f"{name}.pdf",
            "pages_indexed": used_pages,
            "pages_in_file": total_pages,
            "chunks": result["chunks"],
            "ocr_pages": result["ocr_pages"],
            "ingest_seconds": seconds,
            **stats,
            "boilerplate": boilerplate(pages),
        }
        doc_reports.append(info)
        print(
            f"indexed {name}.pdf: {used_pages}/{total_pages} pages, {result['chunks']} chunks, "
            f"ocr={result['ocr_pages']}, {seconds}s"
        )

    problems = validate_qa_set(items, pages_by_doc)
    if problems:
        print("QA set validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"qa set validated: {len(items)} questions on {len(pdfs)} real documents")

    retriever = Retriever(store)
    rows: list[dict] = []
    for item in items:
        contexts = retriever.retrieve(
            item["question"],
            k=args.k,
            retrieve_k=args.retrieve_k,
            cross_encoder=not args.no_ce,
        )
        retrieved = [(chunk.source, chunk.page) for chunk in contexts]
        rank = None
        if item.get("in_scope", True):
            rank = first_hit_rank(contexts, f"{item['doc']}.pdf", item["gold_pages"])

        answer = ""
        keyword_ok = refusal_ok = cites = citation_points_to_gold = claim_support = None
        cited = []
        claims = []
        if not args.no_answers and contexts:
            try:
                answer = answer_question(item["question"], contexts)
            except Exception as exc:  # noqa: BLE001
                answer = f"[API error] {exc}"
            if item.get("in_scope", True):
                keys = [key.lower() for key in item.get("must_contain", [])]
                keyword_ok = any(key in answer.lower() for key in keys) if keys else None
            else:
                refusal_ok = is_refusal(answer)
            cites = bool(CITATION_PATTERN.search(answer))
            cited = cited_blocks(answer, retrieved)
            claims = check_answer(answer, contexts)
            claim_support = support_rate(claims)
            if item.get("in_scope", True):
                gold_doc = f"{item['doc']}.pdf"
                citation_points_to_gold = any(
                    source == gold_doc and page in item["gold_pages"] for source, page in cited
                )

        rows.append(
            {
                "id": item["id"],
                "doc": item.get("doc"),
                "question": item["question"],
                "in_scope": item.get("in_scope", True),
                "gold_pages": item.get("gold_pages", []),
                "retrieved": retrieved,
                "hit_rank": rank,
                "hit_at_k": rank is not None if item.get("in_scope", True) else None,
                "hit_at_1": (rank == 1) if rank is not None else (False if item.get("in_scope", True) else None),
                "reciprocal_rank": (1.0 / rank) if rank else (0.0 if item.get("in_scope", True) else None),
                "answer": answer,
                "answer_chars": len(answer),
                "keyword_ok": keyword_ok,
                "refusal_ok": refusal_ok,
                "cites": cites,
                "cited_blocks": cited,
                "citation_points_to_gold": citation_points_to_gold,
                "claim_support": claim_support,
                "claims": [verdict.as_dict for verdict in claims],
            }
        )
        print(f"{item['id']}: rank={rank} {retrieved}")

    in_scope = [row for row in rows if row["in_scope"]]
    out_scope = [row for row in rows if not row["in_scope"]]
    rrs = [row["reciprocal_rank"] for row in in_scope if row["reciprocal_rank"] is not None]
    claim_scores = [row["claim_support"] for row in in_scope if row.get("claim_support") is not None]

    by_doc = {}
    for doc in sorted({row["doc"] for row in in_scope}):
        doc_rows = [row for row in in_scope if row["doc"] == doc]
        doc_rrs = [row["reciprocal_rank"] for row in doc_rows if row["reciprocal_rank"] is not None]
        by_doc[doc] = {
            "n": len(doc_rows),
            "hit_at_k": rate([row["hit_at_k"] for row in doc_rows]),
            "hit_at_1": rate([row["hit_at_1"] for row in doc_rows]),
            "mrr": round(sum(doc_rrs) / len(doc_rrs), 3) if doc_rrs else None,
        }

    summary = {
        "corpus": {
            "documents": len(pdfs),
            "pages_indexed": sum(info["pages_indexed"] for info in doc_reports),
            "pages_in_files": sum(info["pages_in_file"] for info in doc_reports),
            "chunks": store.count(),
            "page_cap": args.max_pages,
        },
        "settings": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "min_chunk_chars": config.MIN_CHUNK_CHARS,
            "embedding_model": config.EMBEDDING_MODEL,
            "retrieve_k": args.retrieve_k,
            "context_k": args.k,
            "mmr_lambda": config.MMR_LAMBDA,
            "cross_encoder": not args.no_ce,
            "rerank_model": config.RERANK_MODEL if not args.no_ce else None,
            "ocr_enabled": bool(config.OCR_ENABLED),
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.LLM_MODEL,
        },
        "n_questions": len(rows),
        "n_in_scope": len(in_scope),
        "hit_at_k": rate([row["hit_at_k"] for row in in_scope]),
        "hit_at_1": rate([row["hit_at_1"] for row in in_scope]),
        "mrr": round(sum(rrs) / len(rrs), 3) if rrs else None,
        "by_document": by_doc,
        "misses": [
            {"id": row["id"], "doc": row["doc"], "gold_pages": row["gold_pages"], "retrieved": row["retrieved"]}
            for row in in_scope
            if row["hit_rank"] is None
        ],
        "answer_keyword_accuracy": rate([row["keyword_ok"] for row in in_scope]),
        "citation_rate": rate([row["cites"] for row in in_scope]),
        "citation_gold_rate": rate([row["citation_points_to_gold"] for row in in_scope]),
        "claim_support_rate": round(sum(claim_scores) / len(claim_scores), 3) if claim_scores else None,
        "answer_mean_chars": round(sum(row["answer_chars"] for row in in_scope) / len(in_scope)) if in_scope else None,
        "refusal_ok": rate([row["refusal_ok"] for row in out_scope]),
        "api_errors": sum(1 for row in rows if (row.get("answer") or "").startswith("[API error]")),
        "documents": doc_reports,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for key, value in summary.items():
        if key not in {"documents", "by_document", "settings"}:
            print(f"{key}: {value}")
    print("\n=== per document ===")
    for doc, metrics in by_doc.items():
        print(f"{doc}: {metrics}")
    print("\n=== documents ===")
    for info in doc_reports:
        print(info)
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
