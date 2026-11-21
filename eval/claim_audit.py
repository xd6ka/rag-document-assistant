"""Score stored answers per claim, against the blocks they actually cite.

Run after ``run_eval.py``: it rebuilds the same throwaway index, replays the same
retrieval for each question, and grades every cited sentence of the stored answer.
No model is called to generate, so a full audit costs one embedding pass.

    python eval/claim_audit.py                     # newest report in eval/reports
    python eval/claim_audit.py --report eval/reports/latest.json --dump eval/reports/claims.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.citations import check_answer, support_rate  # noqa: E402
from src.ingest import ingest_path  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402


def load_qa(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(ROOT / "eval" / "reports" / "latest.json"))
    parser.add_argument("--qa", default=str(ROOT / "eval" / "qa_set.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "eval" / "corpus"))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--dump", default=str(ROOT / "eval" / "reports" / "claims.json"))
    args = parser.parse_args()

    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = payload["rows"]
    items = {item["id"]: item for item in load_qa(Path(args.qa))}

    workdir = Path(tempfile.mkdtemp(prefix="rag_claims_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="claims")
    for pdf in sorted(Path(args.corpus).glob("*.pdf")):
        ingest_path(pdf, store=store)

    retriever = Retriever(store)
    records = []
    for row in rows:
        item = items.get(row["id"], {})
        contexts = retriever.retrieve(
            item.get("question", ""),
            k=args.k,
            retrieve_k=args.retrieve_k,
            cross_encoder=payload.get("summary", {}).get("settings", {}).get("cross_encoder", True),
        )
        blocks = [(chunk.source, chunk.page) for chunk in contexts]
        if blocks != [tuple(pair) for pair in (row.get("retrieved") or [])]:
            print(f"WARNING: {row['id']} retrieval order differs from the stored report")

        verdicts = check_answer(row.get("answer") or "", contexts)
        records.append(
            {
                "id": row["id"],
                "in_scope": row.get("in_scope", True),
                "gold_pages": row.get("gold_pages", []),
                "gold_doc": row.get("doc"),
                "claim_support": support_rate(verdicts),
                "verdicts": [v.as_dict for v in verdicts],
            }
        )

    graded = [r for r in records if r["in_scope"] and r["claim_support"] is not None]
    mean = round(sum(r["claim_support"] for r in graded) / len(graded), 3) if graded else None
    print(f"\n=== claim-level citation audit: {len(graded)} in-scope answers, mean support {mean} ===")
    for record in records:
        if not record["in_scope"]:
            continue
        flags = [v for v in record["verdicts"] if v["verdict"] in {"misattributed", "weak"}]
        mark = "OK " if not flags else "!! "
        print(f"{mark}{record['id']} support={record['claim_support']} claims={len(record['verdicts'])}")
        for verdict in flags:
            print(
                f"     {verdict['verdict']}: cited {verdict['cited_blocks']} best {verdict['best_block']} "
                f"sim {verdict['cited_similarity']}->{verdict['best_similarity']} "
                f"overlap {verdict['cited_overlap']}->{verdict['best_overlap']}"
            )
            print(f"     text: {verdict['sentence'][:160]}")

    dump = Path(args.dump)
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "report": args.report,
                "mean_claim_support": mean,
                "rows": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nclaims: {dump}")


if __name__ == "__main__":
    main()
