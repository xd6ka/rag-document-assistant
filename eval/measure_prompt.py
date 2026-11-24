"""Measure what the prompt costs: characters in, seconds out.

The streaming section established that a 25-second answer spends ~24 seconds before
the first token, and that on CPU the model is prefilling ~900 tokens of numbered
context, not generating. This script turns that claim into a sweep: the same
questions are answered with the same retrieval at several ``CONTEXT_BUDGET``
values, and each row records the prompt size and the time to first token.

Nothing about the pipeline changes between rows except the number of characters
handed to the model, so the difference is the prefill and nothing else. ``budget=0``
is the shipped configuration.

    python eval/measure_prompt.py
    python eval/measure_prompt.py --budgets 0,400,250 --questions 3
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import load_qa_set  # noqa: E402
from src import config  # noqa: E402
from src.generation import build_prompt, provider_label, stream_answer  # noqa: E402
from src.ingest import ingest_path  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

REPORT = ROOT / "eval" / "reports" / "prompt_budget.json"


def stream_once(question: str, contexts) -> dict:
    """One streamed answer, timed at the first token and at the last."""
    started = time.perf_counter()
    first_token = None
    answer = ""
    for delta in stream_answer(question, contexts):
        if first_token is None:
            first_token = time.perf_counter() - started
        answer += delta
    return {
        "first_token_s": round(first_token, 3) if first_token else None,
        "total_s": round(time.perf_counter() - started, 3),
        "answer": answer.strip(),
        "answer_chars": len(answer.strip()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-size vs time-to-first-token on the synthetic corpus")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval" / "corpus")
    parser.add_argument("--qa", type=Path, default=ROOT / "eval" / "qa_set.jsonl")
    parser.add_argument("--budgets", default="0,400,250", help="characters per block; 0 = whole block")
    parser.add_argument("--k", type=int, default=config.CONTEXT_K)
    parser.add_argument("--retrieve-k", type=int, default=config.RETRIEVE_K)
    parser.add_argument("--questions", type=int, default=3, help="answer only the first N in-scope questions")
    parser.add_argument("--no-ce", action="store_true", help="skip the reranker (not the shipped path)")
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()

    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    if 0 not in budgets:
        budgets.insert(0, 0)

    items = [item for item in load_qa_set(args.qa) if item.get("in_scope", True)][: args.questions]
    pdfs = sorted(args.corpus.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {args.corpus}")

    workdir = Path(tempfile.mkdtemp(prefix="rag_prompt_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="prompt")
    for pdf in pdfs:
        result = ingest_path(pdf, store=store)
        print(f"indexed {pdf.name}: {result['chunks']} chunks")

    retriever = Retriever(store)
    contexts_by_id = {
        item["id"]: retriever.retrieve(
            item["question"],
            k=args.k,
            retrieve_k=args.retrieve_k,
            cross_encoder=not args.no_ce,
        )
        for item in items
    }

    # One throwaway call so the model is resident before the first timed row; Ollama
    # otherwise charges the weights load (~20 s) to whichever budget ran first. The
    # warmup must not share its *context* with a measured row: Ollama caches the
    # longest common prefix, the question sits after the context, and a warmup over
    # the same blocks returns the first row in 2 s instead of 30.
    print("warming up the model…")
    spare = next((item for item in load_qa_set(args.qa) if item["id"] not in {i["id"] for i in items}), None)
    warm_contexts = contexts_by_id[items[0]["id"]] if spare is None else retriever.retrieve(
        spare["question"], k=args.k, retrieve_k=args.retrieve_k, cross_encoder=not args.no_ce
    )
    stream_once(spare["question"] if spare else "Name a document in this collection.", warm_contexts)
    time.sleep(2)

    report: dict = {
        "provider": provider_label(),
        "k": args.k,
        "retrieve_k": args.retrieve_k,
        "cross_encoder": not args.no_ce,
        "questions": [item["id"] for item in items],
        "budgets": {},
    }

    for budget in budgets:
        config.CONTEXT_BUDGET = budget
        rows = []
        for item in items:
            contexts = contexts_by_id[item["id"]]
            prompt = build_prompt(item["question"], contexts)
            row = stream_once(item["question"], contexts)
            keys = [key.lower() for key in item.get("must_contain", [])]
            row.update(
                {
                    "id": item["id"],
                    "prompt_chars": len(prompt),
                    "block_chars": sum(len(chunk.text) for chunk in contexts),
                    "keyword_ok": any(key in row["answer"].lower() for key in keys) if keys else None,
                }
            )
            rows.append(row)
            print(
                f"budget={budget:<4} {item['id']}: prompt {len(prompt)} chars, "
                f"first token {row['first_token_s']} s, total {row['total_s']} s"
            )

        timed = [row for row in rows if row["first_token_s"]]
        keywords = [row["keyword_ok"] for row in rows if row["keyword_ok"] is not None]
        report["budgets"][str(budget)] = {
            "mean_prompt_chars": round(sum(row["prompt_chars"] for row in rows) / len(rows), 1),
            "mean_first_token_s": round(sum(row["first_token_s"] for row in timed) / len(timed), 3) if timed else None,
            "mean_total_s": round(sum(row["total_s"] for row in rows) / len(rows), 3),
            "keyword_rate": round(sum(1 for flag in keywords if flag) / len(keywords), 3) if keywords else None,
            "rows": rows,
        }

    plain = report["budgets"]["0"]
    for key, value in report["budgets"].items():
        if key == "0":
            continue
        value["prompt_chars_vs_full"] = round(1 - value["mean_prompt_chars"] / plain["mean_prompt_chars"], 3)
        if value["mean_first_token_s"] and plain["mean_first_token_s"]:
            value["first_token_saved"] = round(plain["mean_first_token_s"] - value["mean_first_token_s"], 3)

    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nbudget | mean prompt chars | first token | total | keyword")
    for key, value in report["budgets"].items():
        print(
            f"{key:>6} | {value['mean_prompt_chars']:>17} | {value['mean_first_token_s']:>11} s"
            f" | {value['mean_total_s']:>5} s | {value['keyword_rate']}"
        )
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
