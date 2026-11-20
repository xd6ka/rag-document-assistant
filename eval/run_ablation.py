"""Retrieval ablation: which configuration actually retrieves better?

Runs the same labelled questions against several chunking / reranking
configurations and reports hit@1, hit@k and MRR. Retrieval only, no LLM calls,
so it costs nothing and takes seconds.

    python eval/run_ablation.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_eval import first_hit_rank, load_qa_set, read_page_texts  # noqa: E402
from src.chunking import chunk_pages  # noqa: E402
from src.embeddings import embed_texts  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

CONFIGS = [
    {"name": "500/0, no MMR, k=3", "size": 500, "overlap": 0, "rerank": False, "k": 3},
    {"name": "500/50, no MMR, k=3", "size": 500, "overlap": 50, "rerank": False, "k": 3},
    {"name": "800/150, no MMR, k=3", "size": 800, "overlap": 150, "rerank": False, "k": 3},
    {"name": "800/150, MMR, k=3", "size": 800, "overlap": 150, "rerank": True, "k": 3},
    {"name": "800/150, MMR, k=4", "size": 800, "overlap": 150, "rerank": True, "k": 4},
]

RETRIEVE_K = 8


def build_store(pages_by_doc: dict[str, list[str]], size: int, overlap: int, path: Path) -> DocumentStore:
    store = DocumentStore(path=path, collection_name=f"abl_{size}_{overlap}")
    for source, pages in pages_by_doc.items():
        chunks = chunk_pages(pages, size, overlap)
        if not chunks:
            continue
        store.add_document(source, chunks, embed_texts([chunk.text for chunk in chunks]))
    return store


def main() -> None:
    corpus = ROOT / "eval" / "corpus"
    pages_by_doc = read_page_texts(corpus)
    questions = [item for item in load_qa_set(ROOT / "eval" / "qa_set.jsonl") if item.get("in_scope", True)]

    workdir = Path(tempfile.mkdtemp(prefix="rag_ablation_"))
    results = []

    for config in CONFIGS:
        store = build_store(pages_by_doc, config["size"], config["overlap"], workdir / f"c{len(results)}")
        retriever = Retriever(store)

        ranks: list[int | None] = []
        for item in questions:
            contexts = retriever.retrieve(
                item["question"],
                k=config["k"],
                retrieve_k=RETRIEVE_K,
                rerank=config["rerank"],
            )
            ranks.append(first_hit_rank(contexts, f"{item['doc']}.pdf", item["gold_pages"]))

        hits = [rank for rank in ranks if rank is not None]
        hit_at_k = len(hits) / len(questions)
        hit_at_1 = sum(1 for rank in ranks if rank == 1) / len(questions)
        mrr = sum(1.0 / rank for rank in hits) / len(questions)

        results.append(
            {
                "config": config["name"],
                "chunks": store.count(),
                "hit_at_k": round(hit_at_k, 3),
                "hit_at_1": round(hit_at_1, 3),
                "mrr": round(mrr, 3),
                "misses": [item["id"] for item, rank in zip(questions, ranks) if rank is None],
            }
        )
        print(
            f"{config['name']:<24} chunks={store.count():<3} "
            f"hit@{config['k']}={hit_at_k:.3f} hit@1={hit_at_1:.3f} mrr={mrr:.3f}"
        )

    out_path = ROOT / "eval" / "reports" / "ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "n_questions": len(questions),
                "retrieve_k": RETRIEVE_K,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
