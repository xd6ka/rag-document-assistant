"""Multilingual retrieval ablation: does swapping the models fix ru/zh questions?

The English QA set is asked again in Russian and Chinese against English PDFs. The
English pipeline is expected to fail here, and for a reason worth separating:

* the bi-encoder decides *which* eight candidates exist, so an English embedder
  never lets the correct page into the pool — no reranker can rescue it;
* a cross-encoder only reorders what it is given, so swapping it alone cannot
  fix a missing candidate.

Hence five configurations, each changing exactly one thing:

    EN embed                 the shipped pipeline, asked in ru/zh (the baseline)
    EN embed + EN CE         English reranker over English candidates
    ML embed                 multilingual bi-encoder only
    ML embed + EN CE         multilingual candidates, English reranker
    ML embed + ML CE         both sides multilingual

Retrieval only — no LLM calls, nothing to pay for. MMR stays on and k=4 in every
configuration so the models are the only variable.

    python eval/run_multilingual.py
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

from eval.run_eval import first_hit_rank, load_qa_set, read_page_texts, validate_qa_set  # noqa: E402
from src.chunking import chunk_pages  # noqa: E402
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CONTEXT_K  # noqa: E402
from src.embeddings import embed_texts  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

EN_EMBED = "all-MiniLM-L6-v2"
EN_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ML_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ML_CE = "BAAI/bge-reranker-base"

RETRIEVE_K = 8

CONFIGS = [
    {"name": "EN embed", "embed": EN_EMBED, "ce": None},
    {"name": "EN embed + EN CE", "embed": EN_EMBED, "ce": EN_CE},
    {"name": "ML embed", "embed": ML_EMBED, "ce": None},
    {"name": "ML embed + EN CE", "embed": ML_EMBED, "ce": EN_CE},
    {"name": "ML embed + ML CE", "embed": ML_EMBED, "ce": ML_CE},
]


def build_store(
    pages_by_doc: dict[str, list[str]],
    embed_model: str,
    path: Path,
) -> DocumentStore:
    store = DocumentStore(path=path, collection_name="ml_abl")
    for source, pages in pages_by_doc.items():
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue
        store.add_document(
            source,
            chunks,
            embed_texts([chunk.text for chunk in chunks], model_name=embed_model),
        )
    return store


def score(items: list[dict], ranks: list[int | None]) -> dict:
    hits = [rank for rank in ranks if rank is not None]
    if not ranks:
        return {"hit_at_k": None, "hit_at_1": None, "mrr": None, "n_questions": 0}
    return {
        "n_questions": len(ranks),
        "hit_at_k": round(len(hits) / len(ranks), 3),
        "hit_at_1": round(sum(1 for rank in ranks if rank == 1) / len(ranks), 3),
        "mrr": round(sum(1.0 / rank for rank in hits) / len(ranks), 3),
        "misses": [item["id"] for item, rank in zip(items, ranks) if rank is None],
    }


def main() -> None:
    corpus = ROOT / "eval" / "corpus"
    pages_by_doc = read_page_texts(corpus)
    items = load_qa_set(ROOT / "eval" / "qa_set_multilingual.jsonl")

    errors = validate_qa_set(items, pages_by_doc)
    if errors:
        # A drifted label would inflate the numbers silently; fail loudly instead.
        print("qa set validation failed:")
        for error in errors:
            print(f"  {error}")
        raise SystemExit(1)
    print(f"{len(items)} questions validated against the corpus")

    workdir = Path(tempfile.mkdtemp(prefix="rag_ml_ablation_"))
    stores: dict[str, DocumentStore] = {}
    results = []

    for config in CONFIGS:
        embed_model = config["embed"]
        if embed_model not in stores:
            stores[embed_model] = build_store(pages_by_doc, embed_model, workdir / f"s{len(stores)}")
        store = stores[embed_model]
        retriever = Retriever(store)

        ranks: list[int | None] = []
        started = time.perf_counter()
        for item in items:
            contexts = retriever.retrieve(
                item["question"],
                k=CONTEXT_K,
                retrieve_k=RETRIEVE_K,
                rerank=True,
                cross_encoder=bool(config["ce"]),
                embed_model=embed_model,
                rerank_model=config["ce"],
            )
            ranks.append(first_hit_rank(contexts, f"{item['doc']}.pdf", item["gold_pages"]))
        elapsed = time.perf_counter() - started

        overall = score(items, ranks)
        by_language = {}
        for lang in ("ru", "zh"):
            subset = [item for item in items if item["lang"] == lang]
            by_language[lang] = score(subset, [rank for item, rank in zip(items, ranks) if item["lang"] == lang])

        results.append(
            {
                "config": config["name"],
                "embed_model": embed_model,
                "cross_encoder": config["ce"],
                "chunks": store.count(),
                **overall,
                "by_language": by_language,
                "seconds_per_question": round(elapsed / len(items), 3),
            }
        )
        print(
            f"{config['name']:<20} hit@{CONTEXT_K}={overall['hit_at_k']} "
            f"hit@1={overall['hit_at_1']} mrr={overall['mrr']} "
            f"ru={by_language['ru']['hit_at_k']} zh={by_language['zh']['hit_at_k']} "
            f"{elapsed / len(items):.3f}s/q misses={len(overall['misses'])}"
        )

    out_path = ROOT / "eval" / "reports" / "multilingual.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "n_questions": len(items),
                "languages": ["ru", "zh"],
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "retrieve_k": RETRIEVE_K,
                "context_k": CONTEXT_K,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
