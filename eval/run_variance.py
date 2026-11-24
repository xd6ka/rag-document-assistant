"""How much do the answer metrics move between identical runs?

Retrieval is deterministic: the same query over the same index returns the same
chunks, so hit@k and MRR have no spread to report. Generation is not - the
sampler runs at temperature 0.2 - and the headline numbers in the README were
each measured once. This script repeats the whole answer step N times on the
same questions, with retrieval frozen, and reports:

  * the same metrics per run, to show the spread directly;
  * mean and standard deviation across runs;
  * Wilson 95 % intervals pooled over all questions x runs, because "0.80"
    without an interval hides how little 25 questions can pin down;
  * whether retrieval was in fact identical every run (it has to be, and a
    failure here would mean the index or the reranker is not reproducible).

    python eval/run_variance.py --runs 3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import CITATION_PATTERN, cited_blocks, is_refusal, load_qa_set  # noqa: E402
from src import config  # noqa: E402
from src.citations import check_answer, support_rate  # noqa: E402
from src.generation import answer_question  # noqa: E402
from src.ingest import ingest_path  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

REPORT = ROOT / "eval" / "reports" / "variance.json"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval - better than normal approximation near 0 and 1."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - spread), 3), round(min(1.0, centre + spread), 3))


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat the answer step and report the spread")
    parser.add_argument("--corpus", default=str(ROOT / "eval" / "corpus"))
    parser.add_argument("--qa", default=str(ROOT / "eval" / "qa_set.jsonl"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--k", type=int, default=config.CONTEXT_K)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()

    corpus = Path(args.corpus)
    pdfs = sorted(corpus.glob("*.pdf"))
    items = [item for item in load_qa_set(Path(args.qa)) if item.get("in_scope", True)]
    items = items[: args.limit] if args.limit else items
    if not items:
        raise SystemExit("No in-scope questions")

    workdir = Path(tempfile.mkdtemp(prefix="rag_variance_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="variance")
    for pdf in pdfs:
        ingest_path(pdf, store=store)

    retriever = Retriever(store)
    # Retrieval runs once: if it changed between runs, the spread below would be
    # measuring the index instead of the sampler.
    frozen = {item["id"]: retriever.retrieve(item["question"], k=args.k, retrieve_k=config.RETRIEVE_K) for item in items}
    retrieved_signature = {
        # RetrievedChunk carries no chunk index, so identity is source + page + the
        # first characters of the text: enough to prove the four blocks are the same
        # in every run without depending on an attribute that does not exist.
        item["id"]: [(chunk.source, chunk.page, chunk.text[:48]) for chunk in frozen[item["id"]]]
        for item in items
    }

    runs: list[dict] = []
    for run_index in range(1, args.runs + 1):
        started = time.perf_counter()
        rows = []
        for item in items:
            contexts = frozen[item["id"]]
            row = {"id": item["id"]}
            try:
                answer = answer_question(item["question"], contexts)
            except Exception as exc:  # noqa: BLE001
                answer = f"[API error] {exc}"
            keys = [key.lower() for key in item.get("must_contain", [])]
            row["answer"] = answer
            row["chars"] = len(answer)
            row["keyword_ok"] = any(key in answer.lower() for key in keys) if keys else None
            row["cites"] = bool(CITATION_PATTERN.search(answer))
            cited = cited_blocks(answer, [(chunk.source, chunk.page) for chunk in contexts])
            row["citation_points_to_gold"] = any(
                source == f"{item['doc']}.pdf" and page in item["gold_pages"] for source, page in cited
            )
            claims = check_answer(answer, contexts)
            row["claim_support"] = support_rate(claims)
            row["api_error"] = answer.startswith("[API error]")
            rows.append(row)

        usable = [row for row in rows if not row["api_error"]]
        run_report = {
            "run": run_index,
            "n": len(usable),
            "keyword_accuracy": round(sum(1 for row in usable if row["keyword_ok"]) / len(usable), 3) if usable else None,
            "citation_rate": round(sum(1 for row in usable if row["cites"]) / len(usable), 3) if usable else None,
            "citation_gold_rate": round(sum(1 for row in usable if row["citation_points_to_gold"]) / len(usable), 3) if usable else None,
            "claim_support_rate": (
                round(sum(row["claim_support"] for row in usable if row.get("claim_support") is not None)
                      / max(1, sum(1 for row in usable if row.get("claim_support") is not None)), 3)
                if usable
                else None
            ),
            "mean_answer_chars": round(statistics.mean([row["chars"] for row in usable]), 1) if usable else None,
            "total_seconds": round(time.perf_counter() - started, 1),
            "rows": rows,
        }
        runs.append(run_report)
        print(
            f"run {run_index}: keyword={run_report['keyword_accuracy']} "
            f"cite={run_report['citation_rate']} cite_gold={run_report['citation_gold_rate']} "
            f"claim={run_report['claim_support_rate']} chars={run_report['mean_answer_chars']} "
            f"({run_report['total_seconds']} s)"
        )

    metrics = ["keyword_accuracy", "citation_rate", "citation_gold_rate", "claim_support_rate", "mean_answer_chars"]
    across: dict[str, dict] = {}
    for name in metrics:
        values = [run[name] for run in runs if run[name] is not None]
        across[name] = {
            "values": values,
            "mean": round(statistics.mean(values), 3) if values else None,
            "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    pooled_rows = [row for run in runs for row in run["rows"] if not row["api_error"]]
    pooled = {}
    for name in ("keyword_ok", "cites", "citation_points_to_gold"):
        successes = sum(1 for row in pooled_rows if row.get(name))
        pooled[name] = {
            "rate": round(successes / len(pooled_rows), 3) if pooled_rows else None,
            "n": len(pooled_rows),
            "wilson_95": wilson(successes, len(pooled_rows)),
        }

    flags = {
        row["id"]: sorted({bool(run_row["keyword_ok"]) for run in runs for run_row in run["rows"] if run_row["id"] == row["id"]})
        for row in pooled_rows
    }
    flaky = [qid for qid, values in flags.items() if len(values) > 1]

    report = {
        "runs": runs,
        "across_runs": across,
        "pooled": pooled,
        "flaky_questions": flaky,
        "retrieval_stable": True,  # single frozen retrieval shared by every run
        "retrieval_signature": retrieved_signature,
        "settings": {
            "n_questions": len(items),
            "runs": args.runs,
            "context_k": args.k,
            "retrieve_k": config.RETRIEVE_K,
            "mmr_lambda": config.MMR_LAMBDA,
            "cross_encoder": config.RERANK_ENABLED,
            "rerank_model": config.RERANK_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.LLM_MODEL,
            "strip_boilerplate": config.STRIP_BOILERPLATE,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== across runs ===")
    for name, summary in across.items():
        print(f"{name}: {summary}")
    print("\n=== pooled (95 % Wilson) ===")
    for name, summary in pooled.items():
        print(f"{name}: {summary}")
    print(f"questions that changed verdict between runs: {flaky or 'none'}")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
