"""Streaming, measured: when does the first word arrive, and when does it finish?

Two facts are easy to conflate here, so this measures them separately:

* streaming does not make a local model faster — the total wall time is the same;
* on CPU the wait is dominated by prompt prefill, not by token generation, so the
  first token does not arrive nearly as early as "streaming" suggests.

Every request carries a marker in the *first context block*, because Ollama caches
the longest common prefix of a prompt: a marker at the end of the question leaves
the whole context cached, and the second call then returns in ~1.5 s — measuring
the cache, not the model.

    python eval/measure_stream.py           # 3 questions, both modes
    python eval/measure_stream.py --limit 5

Requires the generation backend from .env (local Ollama by default).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_eval import load_qa_set, read_page_texts  # noqa: E402
from src import config  # noqa: E402
from src.chunking import chunk_pages  # noqa: E402
from src.embeddings import embed_texts  # noqa: E402
from src.generation import answer_question, provider_label, stream_answer  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore, RetrievedChunk  # noqa: E402

DEFAULT_IDS = ["e01", "v01", "d02"]
WARMUP_QUESTION = "Warm up the model, one token only."


def build_store(pages_by_doc: dict[str, list[str]], path: Path) -> DocumentStore:
    store = DocumentStore(path=path, collection_name="stream_eval")
    for source, pages in pages_by_doc.items():
        chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if chunks:
            store.add_document(source, chunks, embed_texts([chunk.text for chunk in chunks]))
    return store


def mark(contexts: list[RetrievedChunk], tag: str) -> list[RetrievedChunk]:
    """Change the first block's source so the engine cannot reuse a cached prefix."""
    if not contexts:
        return contexts
    return [
        replace(contexts[0], source=f"{contexts[0].source}{tag}"),
        *contexts[1:],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(DEFAULT_IDS))
    args = parser.parse_args()

    items = {item["id"]: item for item in load_qa_set(ROOT / "eval" / "qa_set.jsonl") if item.get("in_scope", True)}
    selected = [items[question_id] for question_id in DEFAULT_IDS[: args.limit]]

    store = build_store(read_page_texts(ROOT / "eval" / "corpus"), Path(tempfile.mkdtemp(prefix="rag_stream_")))
    retriever = Retriever(store)

    # Warm-up moves the weights into RAM so the first measured question does not
    # carry a ~30 s model load.
    contexts = retriever.retrieve(selected[0]["question"], rerank=True, cross_encoder=True)
    started = time.perf_counter()
    list(stream_answer(WARMUP_QUESTION, mark(contexts, "#warm"), max_tokens=1))
    warmup = time.perf_counter() - started
    print(f"warm-up (model load) {warmup:.1f}s, provider {provider_label()}")

    rows = []
    for index, item in enumerate(selected):
        base_contexts = retriever.retrieve(item["question"], rerank=True, cross_encoder=True)

        stream_contexts = mark(base_contexts, f"#stream{index}")
        started = time.perf_counter()
        first_token = None
        pieces: list[str] = []
        for delta in stream_answer(item["question"], stream_contexts):
            if first_token is None:
                first_token = time.perf_counter() - started
            pieces.append(delta)
        stream_total = time.perf_counter() - started
        stream_text = "".join(pieces).strip()

        blocking_contexts = mark(base_contexts, f"#block{index}")
        started = time.perf_counter()
        blocking_text = answer_question(item["question"], blocking_contexts)
        blocking_total = time.perf_counter() - started

        rows.append(
            {
                "id": item["id"],
                "ttft_seconds": round(first_token, 2) if first_token is not None else None,
                "stream_total_seconds": round(stream_total, 2),
                "blocking_total_seconds": round(blocking_total, 2),
                "chars": len(stream_text),
                "same_text": stream_text == blocking_text.strip(),
            }
        )
        print(
            f"{item['id']}: first token {rows[-1]['ttft_seconds']}s · streamed {stream_total:.1f}s · "
            f"blocking {blocking_total:.1f}s · {len(stream_text)} chars · identical={rows[-1]['same_text']}"
        )

    out_path = ROOT / "eval" / "reports" / "streaming.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "provider": provider_label(),
                "warmup_seconds": round(warmup, 1),
                "rows": rows,
                "mean_ttft_seconds": round(sum(r["ttft_seconds"] for r in rows) / len(rows), 1),
                "mean_stream_total_seconds": round(sum(r["stream_total_seconds"] for r in rows) / len(rows), 1),
                "mean_blocking_total_seconds": round(sum(r["blocking_total_seconds"] for r in rows) / len(rows), 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
