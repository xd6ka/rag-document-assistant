"""Evaluate retrieval and answer quality on the synthetic corpus.

    python eval/build_corpus.py              # (re)generate the PDFs
    python eval/run_eval.py                  # retrieval + answers (uses the API)
    python eval/run_eval.py --no-answers     # retrieval metrics only, no API cost
    python eval/run_eval.py --judge          # add LLM faithfulness scoring

The corpus is indexed into a throwaway store, so results never depend on what
is inside the working collection. Before scoring, every gold label is verified
against the actual page text; a drifted rubric fails loudly instead of quietly
inflating the numbers.
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

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.citations import check_answer, support_rate  # noqa: E402
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
    "unable to",
    "does not provide",
    "doesn't provide",
    "not provided",
    "no relevant",
    "не знаю",
    "не содержит",
)

# Marker lists miss legitimate phrasings ("does not provide information in the
# given context" scored as a failure once), so a negation pattern backs them up.
REFUSAL_PATTERN = re.compile(
    r"(?:does|do|did)\s+not\s+(?:contain|provide|mention|include|cover|address)"
    r"|cannot\s+(?:find|answer|determine|tell)"
    r"|can't\s+(?:find|answer|determine|tell)"
    r"|unable\s+to"
    r"|no\s+(?:information|mention|reference)"
    r"|not\s+(?:in|stated\s+in)\s+the\s+(?:context|document|text)",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """True when the answer says the context does not hold the answer.

    Lexical on purpose: a small local model paraphrases, and a judge here is
    unreliable, so the check errs towards matching many phrasings.
    """
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS) or bool(REFUSAL_PATTERN.search(answer))

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

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


def read_page_texts(corpus_dir: Path) -> dict[str, list[str]]:
    """Extract per-page text for every corpus PDF, keyed by file name."""
    pages_by_doc: dict[str, list[str]] = {}
    for pdf in sorted(corpus_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        pages_by_doc[pdf.name] = [(page.extract_text() or "") for page in reader.pages]
    return pages_by_doc


def validate_qa_set(items: list[dict], pages_by_doc: dict[str, list[str]]) -> list[str]:
    """Check that every gold label points at text that really contains the fact."""
    errors: list[str] = []
    for item in items:
        if not item.get("in_scope", True):
            continue

        doc = item.get("doc")
        gold_pages = item.get("gold_pages") or []
        if not doc or not gold_pages:
            errors.append(f"{item['id']}: in-scope item without doc or gold_pages")
            continue

        source = f"{doc}.pdf"
        pages = pages_by_doc.get(source)
        if pages is None:
            errors.append(f"{item['id']}: unknown document {source}")
            continue

        for page in gold_pages:
            if page < 1 or page > len(pages):
                errors.append(f"{item['id']}: page {page} does not exist in {source}")

        token = item.get("page_contains")
        if token:
            haystack = " ".join(pages[page - 1] for page in gold_pages if 1 <= page <= len(pages))
            if token.lower() not in haystack.lower():
                errors.append(f"{item['id']}: '{token}' not found on gold pages of {source}")

    return errors


def first_hit_rank(contexts, gold_source: str, gold_pages: list[int]) -> int | None:
    for rank, chunk in enumerate(contexts, start=1):
        if chunk.source == gold_source and chunk.page in gold_pages:
            return rank
    return None


def judge_faithfulness(question: str, answer: str, contexts) -> int | None:
    client = get_client()
    context = "\n\n".join(chunk.text for chunk in contexts)
    payload = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        max_tokens=8,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": payload},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    match = re.search(r"[1-5]", text)
    return int(match.group(0)) if match else None


def parse_citations(answer: str) -> list[int]:
    """Block numbers the answer pointed at, in order of appearance."""
    return [int(match) for match in CITATION_PATTERN.findall(answer)]


def cited_blocks(answer: str, retrieved: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Map ``[n]`` markers onto the blocks that were actually in the prompt."""
    return [retrieved[index - 1] for index in parse_citations(answer) if 1 <= index <= len(retrieved)]


def summarize(rows: list[dict], base: dict, no_answers: bool) -> dict:
    """Aggregate per-question rows into the report summary."""
    in_scope = [row for row in rows if row["in_scope"]]
    out_scope = [row for row in rows if not row["in_scope"]]

    def rate(values) -> float | None:
        usable = [v for v in values if v is not None]
        return round(sum(1 for v in usable if v) / len(usable), 3) if usable else None

    rrs = [row["reciprocal_rank"] for row in in_scope if row["reciprocal_rank"] is not None]
    judges = [row["judge"] for row in rows if row["judge"] is not None]
    claim_scores = [row.get("claim_support") for row in in_scope if row.get("claim_support") is not None]
    api_errors = sum(1 for row in rows if (row.get("answer") or "").startswith("[API error]"))

    summary = dict(base)
    summary.update(
        {
            "n_questions": len(rows),
            "n_in_scope": len(in_scope),
            "hit_at_k": rate([row["hit_at_k"] for row in in_scope]),
            "hit_at_1": rate([row["hit_at_1"] for row in in_scope]),
            "mrr": round(sum(rrs) / len(rrs), 3) if rrs else None,
            "answer_keyword_accuracy": rate([row["keyword_ok"] for row in in_scope]),
            "citation_rate": rate([row["cites"] for row in in_scope]),
            # Stronger than citation_rate: does the bracket point at the gold page,
            # or merely at some block? A presence check hides citation drift.
            "citation_gold_rate": rate([row.get("citation_points_to_gold") for row in in_scope]),
            # Weakest link after the gold-page check: each cited *sentence* must be
            # attributable to the block it names, not just to the right page.
            "claim_support_rate": round(sum(claim_scores) / len(claim_scores), 3) if claim_scores else None,
            "refusal_rate": rate([row["refusal_ok"] for row in out_scope]),
            "judge_mean_faithfulness": round(sum(judges) / len(judges), 2) if judges else None,
            # A judge running on the same small local model is not calibrated: see
            # eval/reports/judge_experiment.json, where it scored 7 grounded answers 1/5.
            "judge_reliable": None if not judges else (not config.LLM_IS_LOCAL),
            "api_errors": api_errors,
            # Answer-level metrics are only meaningful when the model actually answered.
            "answer_metrics_valid": None if no_answers else api_errors == 0,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return summary


def rescore_report(src: Path, out: Path, qa_items: list[dict]) -> None:
    """Re-derive answer-level metrics from stored answers, without calling the model.

    Metric definitions change (a refusal marker list, say) long after the answers
    were generated; regenerating 29 answers to restate the same text wastes
    minutes of CPU. This path re-scores what is already in the report.
    """
    payload = json.loads(src.read_text(encoding="utf-8"))
    rows = payload["rows"]
    by_id = {item["id"]: item for item in qa_items}

    for row in rows:
        item = by_id.get(row["id"], {})
        answer = row.get("answer") or ""
        if not answer or answer.startswith("[API error]"):
            continue
        if item.get("in_scope", True):
            keys = [key.lower() for key in item.get("must_contain", [])]
            row["keyword_ok"] = any(key in answer.lower() for key in keys) if keys else None
            row["refusal_ok"] = None
        else:
            row["refusal_ok"] = is_refusal(answer)
            row["keyword_ok"] = None
        row["cites"] = bool(CITATION_PATTERN.search(answer))

        retrieved = [tuple(pair) for pair in (row.get("retrieved") or [])]
        row["cited_blocks"] = cited_blocks(answer, retrieved)
        if item.get("in_scope", True):
            gold_doc = f"{item['doc']}.pdf"
            row["citation_points_to_gold"] = any(
                source == gold_doc and page in item["gold_pages"] for source, page in row["cited_blocks"]
            )
        else:
            row["citation_points_to_gold"] = None

    old = payload.get("summary", {})
    summary = summarize(rows, {"corpus": old.get("corpus"), "settings": old.get("settings")}, no_answers=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== rescored summary ===")
    for key, value in summary.items():
        if key not in {"corpus", "settings"}:
            print(f"{key}: {value}")
    print(f"\nreport: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation set")
    parser.add_argument("--corpus", default=str(ROOT / "eval" / "corpus"), help="directory of corpus PDFs")
    parser.add_argument("--qa", default=str(ROOT / "eval" / "qa_set.jsonl"))
    parser.add_argument("--k", type=int, default=config.CONTEXT_K)
    parser.add_argument("--retrieve-k", type=int, default=config.RETRIEVE_K)
    parser.add_argument("--no-answers", action="store_true", help="skip LLM answer generation")
    parser.add_argument("--judge", action="store_true", help="score answers with an LLM judge")
    parser.add_argument("--no-ce", action="store_true", help="disable cross-encoder reranking")
    parser.add_argument("--limit", type=int, default=0, help="stop after N questions (0 = all)")
    parser.add_argument("--rescore", default=None, help="re-score a previous report's stored answers, no model calls")
    parser.add_argument("--out", default=None, help="report path (default: eval/reports/...)")
    args = parser.parse_args()

    if args.rescore:
        items = load_qa_set(Path(args.qa))
        rescore_report(
            Path(args.rescore),
            Path(args.out) if args.out else Path(args.rescore),
            items,
        )
        return

    corpus_dir = Path(args.corpus)
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {corpus_dir}. Run: python eval/build_corpus.py")

    items = load_qa_set(Path(args.qa))
    if args.limit:
        items = items[: args.limit]
    pages_by_doc = read_page_texts(corpus_dir)

    problems = validate_qa_set(items, pages_by_doc)
    if problems:
        print("QA set validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"qa set validated: {len(items)} questions, {len(pdfs)} documents")

    workdir = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="eval")
    for pdf in pdfs:
        result = ingest_path(pdf, store=store)
        print(f"indexed {result['source']}: {result['chunks']} chunks / {result['pages']} pages")

    retriever = Retriever(store)
    rows: list[dict] = []
    out_path = Path(args.out) if args.out else ROOT / "eval" / "reports" / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for item in items:
        started = time.perf_counter()
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
        keyword_ok = None
        refusal_ok = None
        cites = None
        citation_points_to_gold = None
        cited = []
        claims = []
        claim_support = None
        judge = None

        if not args.no_answers and contexts:
            try:
                answer = answer_question(item["question"], contexts)
            except Exception as exc:  # noqa: BLE001 - keep the run going
                answer = f"[API error] {exc}"

            lowered = answer.lower()
            if item.get("in_scope", True):
                keys = [key.lower() for key in item.get("must_contain", [])]
                keyword_ok = any(key in lowered for key in keys) if keys else None
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

            if args.judge and item.get("in_scope", True) and answer and not answer.startswith("[API error]"):
                try:
                    judge = judge_faithfulness(item["question"], answer, contexts)
                except Exception:
                    judge = None

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
                "keyword_ok": keyword_ok,
                "refusal_ok": refusal_ok,
                "cites": cites,
                "cited_blocks": cited,
                "citation_points_to_gold": citation_points_to_gold,
                "claim_support": claim_support,
                "claims": [verdict.as_dict for verdict in claims],
                "judge": judge,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        print(f"{item['id']}: rank={rank} retrieved={retrieved}")

        # A 25-minute run must not be lost to a later crash; keep a recoverable copy.
        (out_path.parent / "_partial.json").write_text(
            json.dumps({"summary": {}, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = summarize(
        rows,
        {
            "corpus": {
                "documents": len(pdfs),
                "pages": sum(len(pages) for pages in pages_by_doc.values()),
                "chunks": store.count(),
            },
            "settings": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "embedding_model": config.EMBEDDING_MODEL,
                "retrieve_k": args.retrieve_k,
                "context_k": args.k,
                "mmr_lambda": config.MMR_LAMBDA,
                "cross_encoder": not args.no_ce,
                "rerank_model": config.RERANK_MODEL if not args.no_ce else None,
                "llm_provider": config.LLM_PROVIDER,
                "llm_model": config.LLM_MODEL,
            },
        },
        no_answers=args.no_answers,
    )
    judges = [row["judge"] for row in rows if row["judge"] is not None]
    api_errors = summary.get("api_errors") or 0

    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    partial = out_path.parent / "_partial.json"
    if partial.exists():
        partial.unlink()

    if judges and config.LLM_IS_LOCAL:
        print(
            "\nWARNING: the judge shares the small local model with the answerer. "
            "Its score is NOT a calibrated faithfulness metric and must not be reported as one. "
            "Use a hosted model for judging, or audit answers by hand (see README)."
        )

    if api_errors:
        print(
            f"\nWARNING: {api_errors} of {len(rows)} answers failed with API errors. "
            "Answer-level metrics (accuracy, citation rate, refusal, judge) are NOT valid. "
            f"Current endpoint: {config.LLM_PROVIDER} ({config.LLM_BASE_URL}, model {config.LLM_MODEL}). "
            "Check the key, the model name and the account quota, then rerun."
        )

    print("\n=== summary ===")
    for key, value in summary.items():
        if key not in {"corpus", "settings"}:
            print(f"{key}: {value}")
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
