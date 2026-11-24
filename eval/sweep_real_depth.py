"""Where does retrieval break on real PDFs: the vector stage or the reranker?

The first real-corpus run missed 3 of 12 questions. A chunk that never enters
the candidate set cannot be reranked, so the misses could come from stage 1
(depth) or from stage 2 (ordering). This script indexes the real corpus once and
then:

  * retrieves the top 64 candidates by embedding similarity and records where the
    gold page's chunk sits in that list - stage 1 recall;
  * sweeps ``RETRIEVE_K`` (8 / 16 / 32) with the cross-encoder on, to see whether
    hit@4 recovers simply by giving the reranker more candidates;
  * runs retrieve_k=8 without the cross-encoder, as the baseline the reranker
    was supposed to beat.

    python eval/sweep_real_depth.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT_DEPTH = 64

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import load_qa_set  # noqa: E402
from run_real import boilerplate, page_texts, rate, truncate  # noqa: E402
from src import config  # noqa: E402
from src.embeddings import embed_query  # noqa: E402
from src.ingest import ingest_pdf  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

REPORT = ROOT / "eval" / "reports" / "real_depth.json"


def gold_rank(candidates, gold_doc: str, gold_pages: list[int]) -> int | None:
    for rank, chunk in enumerate(candidates, start=1):
        if chunk.source == gold_doc and chunk.page in gold_pages:
            return rank
    return None


def score_group(rows: list[dict]) -> dict:
    rrs = [row["reciprocal_rank"] for row in rows if row["reciprocal_rank"] is not None]
    return {
        "hit_at_k": rate([row["hit_at_k"] for row in rows]),
        "hit_at_1": rate([row["hit_at_1"] for row in rows]),
        "mrr": round(sum(rrs) / len(rrs), 3) if rrs else None,
        "misses": [row["id"] for row in rows if not row["hit_at_k"]],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Depth + reranker sweep on the real corpus")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()
    report_path = Path(args.out)

    corpus = ROOT / "eval" / "real_corpus"
    pdfs = sorted(corpus.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs. Run: python eval/fetch_real_corpus.py")

    items = [item for item in load_qa_set(ROOT / "eval" / "qa_set_real.jsonl") if item.get("in_scope", True)]

    workdir = Path(tempfile.mkdtemp(prefix="rag_depth_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="depth")
    for pdf in pdfs:
        raw, used, total = truncate(pdf, 60)
        result = ingest_pdf(pdf.name, raw, store=store)
        pages = page_texts(raw)
        print(
            f"{pdf.name}: {result['chunks']} chunks ({used}/{total} pages), "
            f"ocr={result['ocr_pages']}, boilerplate={boilerplate(pages)['page_share']}"
        )

    retriever = Retriever(store)
    configs = [
        ("retrieve_k=8, CE on", dict(k=config.CONTEXT_K, retrieve_k=8, cross_encoder=True)),
        ("retrieve_k=16, CE on", dict(k=config.CONTEXT_K, retrieve_k=16, cross_encoder=True)),
        ("retrieve_k=32, CE on", dict(k=config.CONTEXT_K, retrieve_k=32, cross_encoder=True)),
        ("retrieve_k=8, CE off", dict(k=config.CONTEXT_K, retrieve_k=8, cross_encoder=False)),
        ("retrieve_k=32, CE off", dict(k=config.CONTEXT_K, retrieve_k=32, cross_encoder=False)),
    ]

    report: dict = {"k": config.CONTEXT_K, "configs": {}, "stage1_recall": {}}
    for name, kwargs in configs:
        rows = []
        for item in items:
            contexts = retriever.retrieve(item["question"], **kwargs)
            rank = gold_rank(contexts, f"{item['doc']}.pdf", item["gold_pages"])
            rows.append(
                {
                    "id": item["id"],
                    "hit_at_k": rank is not None,
                    "hit_at_1": rank == 1,
                    "reciprocal_rank": (1.0 / rank) if rank else 0.0,
                }
            )
        report["configs"][name] = score_group(rows)
        print(f"{name}: {report['configs'][name]}")

    # Stage 1 alone: how deep must the candidate list go before the gold chunk appears?
    depths = []
    for item in items:
        candidates = store.query(embed_query(item["question"]), ROOT_DEPTH, sources=None)
        rank = gold_rank(candidates, f"{item['doc']}.pdf", item["gold_pages"])
        depths.append(rank)
        print(f"  {item['id']}: gold at vector rank {rank}")
    report["stage1_recall"] = {
        "cutoffs": {str(cut): rate([1 if (rank is not None and rank <= cut) else 0 for rank in depths]) for cut in (8, 16, 32, 64)},
        "ranks": {item["id"]: rank for item, rank in zip(items, depths)},
    }
    print("stage-1 recall:", report["stage1_recall"]["cutoffs"])

    report["corpus"] = {
        "documents": len(pdfs),
        "chunks": store.count(),
        "settings": {
            "embedding_model": config.EMBEDDING_MODEL,
            "rerank_model": config.RERANK_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "mmr_lambda": config.MMR_LAMBDA,
            "page_cap": 60,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
