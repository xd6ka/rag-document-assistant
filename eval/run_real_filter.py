"""Does scoping a question to one document actually help, and by how much?

The UI lets a user restrict a question to a chosen PDF, and `DocumentStore.query`
takes `sources` to do exactly that in the vector search. That feature has never
been measured, so this script measures it on the real corpus, in the honest
version: the filter is set to the document the answer really comes from. That is
the case where the user knows where to look, and it is an *upper bound* — a user
who guesses the wrong document gets nothing, because the filter cannot be
recovered from.

Two numbers are reported per mode, because they answer different questions:

* stage-1 recall — is the gold page inside the ``RETRIEVE_K`` candidates the
  vector index returns, before the cross-encoder runs?
* hit@k / hit@1 / MRR — what survives the full pipeline (CE → MMR → k=4).

    python eval/run_real_filter.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import load_qa_set, validate_qa_set  # noqa: E402
from run_real import PAGE_CAP, page_texts, truncate  # noqa: E402
from src import config  # noqa: E402
from src.embeddings import embed_query  # noqa: E402
from src.ingest import ingest_pdf  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

CORPUS_DIR = ROOT / "eval" / "real_corpus"


def first_hit_rank(contexts, gold_doc: str, gold_pages: list[int]) -> int | None:
    for rank, chunk in enumerate(contexts, start=1):
        if chunk.source == gold_doc and chunk.page in gold_pages:
            return rank
    return None


def metrics(items: list[dict], ranks: list[int | None], gold_hits: list[bool]) -> dict:
    hits = [rank for rank in ranks if rank is not None]
    n = len(items)
    rrs = [1.0 / rank for rank in ranks if rank is not None]
    return {
        "n_questions": n,
        "stage1_recall": round(sum(1 for hit in gold_hits if hit) / n, 3) if n else None,
        "hit_at_k": round(len(hits) / n, 3) if n else None,
        "hit_at_1": round(sum(1 for rank in ranks if rank == 1) / n, 3) if n else None,
        "mrr": round(sum(rrs) / n, 3) if n else None,
        "misses": [item["id"] for item, rank in zip(items, ranks) if rank is None],
    }


def evaluate(retriever: Retriever, items: list[dict], scope: bool) -> dict:
    """One pass over the questions, with or without a per-document filter."""
    ranks: list[int | None] = []
    gold_hits: list[bool] = []
    started = time.perf_counter()

    for item in items:
        gold_doc = f"{item['doc']}.pdf"
        sources = [gold_doc] if scope else None

        # Stage 1 alone: the pool the cross-encoder is allowed to reorder.
        pool = retriever.store.query(
            embed_query(item["question"]),
            config.RETRIEVE_K,
            sources=sources,
        )
        gold_hits.append(first_hit_rank(pool, gold_doc, item["gold_pages"]) is not None)

        contexts = retriever.retrieve(
            item["question"],
            k=config.CONTEXT_K,
            retrieve_k=config.RETRIEVE_K,
            sources=sources,
            cross_encoder=True,
        )
        ranks.append(first_hit_rank(contexts, gold_doc, item["gold_pages"]))

    elapsed = time.perf_counter() - started
    result = metrics(items, ranks, gold_hits)
    result["seconds_per_question"] = round(elapsed / len(items), 3)
    return result


def main() -> None:
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {CORPUS_DIR}. Run: python eval/fetch_real_corpus.py")

    items = [item for item in load_qa_set(ROOT / "eval" / "qa_set_real.jsonl") if item.get("in_scope", True)]
    items = [
        item
        for item in items
        if all(page <= PAGE_CAP for page in (item.get("gold_pages") or []))
    ]

    workdir = Path(tempfile.mkdtemp(prefix="rag_filter_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="filter")
    pages_by_doc: dict[str, list[str]] = {}

    for pdf in pdfs:
        raw, _used, _total = truncate(pdf, PAGE_CAP)
        pages_by_doc[f"{pdf.stem}.pdf"] = page_texts(raw)
        ingest_pdf(f"{pdf.stem}.pdf", raw, store=store)
    print(f"indexed {len(pdfs)} documents, {store.count()} chunks")

    problems = validate_qa_set(items, pages_by_doc)
    if problems:
        print("QA set validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    retriever = Retriever(store)
    results = []
    for scope, label in ((False, "no filter"), (True, "document-scoped")):
        result = evaluate(retriever, items, scope)
        result["mode"] = label
        results.append(result)
        print(
            f"{label:<16} stage1={result['stage1_recall']} hit@{config.CONTEXT_K}={result['hit_at_k']} "
            f"hit@1={result['hit_at_1']} mrr={result['mrr']} {result['seconds_per_question']}s/q "
            f"misses={len(result['misses'])}"
        )

    out = ROOT / "eval" / "reports" / "real_filter.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "corpus": {
                    "documents": len(pdfs),
                    "chunks": store.count(),
                    "page_cap": PAGE_CAP,
                },
                "settings": {
                    "retrieve_k": config.RETRIEVE_K,
                    "context_k": config.CONTEXT_K,
                    "mmr_lambda": config.MMR_LAMBDA,
                    "cross_encoder": config.RERANK_ENABLED,
                    "rerank_model": config.RERANK_MODEL,
                    "embedding_model": config.EMBEDDING_MODEL,
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
