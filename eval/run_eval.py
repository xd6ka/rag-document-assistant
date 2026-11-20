"""Evaluate retrieval and answer quality on a small hand-written question set.

    python eval/run_eval.py                 # retrieval + answers (uses the API)
    python eval/run_eval.py --no-answers    # retrieval metrics only, no API calls
    python eval/run_eval.py --judge         # add LLM faithfulness scoring

The document is indexed into a throwaway store, so results never depend on what
is currently inside your working collection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.generation import answer_question, get_client  # noqa: E402
from src.ingest import ingest_path  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

REFUSAL_MARKERS = (
    "don't know",
    "do not know",
    "not contain",
    "no information",
    "cannot find",
    "can't find",
    "not mentioned",
    "doesn't mention",
    "no mention",
    "not in the context",
    "не знаю",
)

CITATION_PATTERN = re.compile(r"\[\d+\]")

JUDGE_PROMPT = (
    "You are grading a retrieval-augmented answer. Given the CONTEXT and the ANSWER, "
    "rate from 1 to 5 how well the answer is supported by the context only "
    "(5 = every claim is grounded in the context, 1 = the answer hallucinates). "
    "Reply with a single digit and nothing else."
)


def load_qa_set(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def judge_faithfulness(question: str, answer: str, contexts) -> int | None:
    client = get_client()
    context = "\n\n".join(chunk.text for chunk in contexts)
    payload = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"
    response = client.chat.completions.create(
        model=config.QWEN_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": payload},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    match = re.search(r"[1-5]", text)
    return int(match.group(0)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation set")
    parser.add_argument("--pdf", default=str(ROOT / "data" / "real.pdf"))
    parser.add_argument("--qa", default=str(ROOT / "eval" / "qa_set.jsonl"))
    parser.add_argument("--k", type=int, default=config.CONTEXT_K)
    parser.add_argument("--retrieve-k", type=int, default=config.RETRIEVE_K)
    parser.add_argument("--no-answers", action="store_true", help="skip LLM answer generation")
    parser.add_argument("--judge", action="store_true", help="score answers with an LLM judge")
    parser.add_argument("--out", default=None, help="report path (default: eval/reports/...)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    workdir = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="eval")
    ingest = ingest_path(pdf_path, store=store)
    print(f"indexed {ingest['source']}: {ingest['chunks']} chunks / {ingest['pages']} pages")

    retriever = Retriever(store)
    items = load_qa_set(Path(args.qa))

    rows: list[dict] = []
    for item in items:
        started = time.perf_counter()
        contexts = retriever.retrieve(item["question"], k=args.k, retrieve_k=args.retrieve_k)
        retrieved_pages = [chunk.page for chunk in contexts]
        gold = item.get("gold_pages", [])
        hit = any(page in retrieved_pages for page in gold) if gold else None
        hit_at_1 = (retrieved_pages[0] in gold) if (gold and retrieved_pages) else None

        answer = ""
        keyword_ok: bool | None = None
        refusal_ok: bool | None = None
        cites: bool | None = None
        judge: int | None = None

        if not args.no_answers and contexts:
            try:
                answer = answer_question(item["question"], contexts)
            except Exception as exc:  # noqa: BLE001 - keep the run going
                answer = f"[API error] {exc}"

            lowered = answer.lower()
            if item.get("in_scope", True):
                keys = [k.lower() for k in item.get("must_contain", [])]
                keyword_ok = any(k in lowered for k in keys) if keys else None
            else:
                refusal_ok = any(marker in lowered for marker in REFUSAL_MARKERS)
            cites = bool(CITATION_PATTERN.search(answer))

            if args.judge and item.get("in_scope", True) and answer:
                try:
                    judge = judge_faithfulness(item["question"], answer, contexts)
                except Exception:
                    judge = None

        rows.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "in_scope": item.get("in_scope", True),
                "gold_pages": gold,
                "retrieved_pages": retrieved_pages,
                "hit": hit,
                "hit_at_1": hit_at_1,
                "scores": [round(chunk.score, 3) for chunk in contexts],
                "answer": answer,
                "keyword_ok": keyword_ok,
                "refusal_ok": refusal_ok,
                "cites": cites,
                "judge": judge,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        print(f"{item.get('id')}: hit={hit} pages={retrieved_pages}")

    in_scope = [row for row in rows if row["in_scope"]]
    out_scope = [row for row in rows if not row["in_scope"]]

    def rate(values: list[bool | None]) -> float | None:
        usable = [v for v in values if v is not None]
        return round(sum(1 for v in usable if v) / len(usable), 3) if usable else None

    judges = [row["judge"] for row in rows if row["judge"] is not None]
    summary = {
        "documents": {ingest["source"]: {"chunks": ingest["chunks"], "pages": ingest["pages"]}},
        "settings": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "embedding_model": config.EMBEDDING_MODEL,
            "retrieve_k": args.retrieve_k,
            "context_k": args.k,
            "mmr_lambda": config.MMR_LAMBDA,
            "llm_model": config.QWEN_MODEL,
        },
        "n_questions": len(rows),
        "n_in_scope": len(in_scope),
        "retrieval_hit_at_k": rate([row["hit"] for row in in_scope]),
        "retrieval_hit_at_1": rate([row["hit_at_1"] for row in in_scope]),
        "answer_keyword_accuracy": rate([row["keyword_ok"] for row in in_scope]),
        "citation_rate": rate([row["cites"] for row in in_scope]),
        "refusal_rate": rate([row["refusal_ok"] for row in out_scope]),
        "judge_mean_faithfulness": round(sum(judges) / len(judges), 2) if judges else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    out_path = Path(args.out) if args.out else ROOT / "eval" / "reports" / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for key, value in summary.items():
        if key not in {"documents", "settings"}:
            print(f"{key}: {value}")
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
