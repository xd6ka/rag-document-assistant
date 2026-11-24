"""Score sentence-level entailment on the real-corpus answers.

The threshold in :mod:`src.config` was chosen on the 25 hand-audited claims of the
synthetic corpus, which makes every number measured on that set an *in-sample*
number. This script takes the calibration out of sample: it re-scores the claims of
``eval/reports/real_corpus.json`` against the blocks they cite in the ten-document
corpus. There are no hand labels here, so the output is a list of flagged claims to
audit by eye, not an accuracy score — printing the block text next to the claim is
the whole point.

Retrieval is replayed, not invented: the same ``truncate``/``ingest_pdf`` path as
``eval/run_real.py`` runs here, and a row is marked ``replayed`` only when the
retrieved (source, page) pairs match what the stored report recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.entailment import entail_max, is_meta_claim  # noqa: E402
from src.ingest import ingest_pdf  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

PAGE_CAP = 60


def truncate(pdf_path: Path, max_pages: int) -> bytes:
    """PDF bytes limited to the first ``max_pages`` pages, as ``run_real.py`` does."""
    reader = PdfReader(str(pdf_path))
    if not max_pages or len(reader.pages) <= max_pages:
        return pdf_path.read_bytes()
    writer = PdfWriter()
    for page in reader.pages[:max_pages]:
        writer.add_page(page)
    buffer = tempfile.SpooledTemporaryFile()
    writer.write(buffer)
    buffer.seek(0)
    return buffer.read()


def build_store(corpus: Path, index_dir: Path | None) -> tuple[DocumentStore, bool]:
    """Index the corpus, or reuse a cached index when one is already there."""
    if index_dir is not None:
        store = DocumentStore(path=index_dir, collection_name="real")
        count = store.count()
        if count:
            return store, True
    else:
        store = DocumentStore(path=Path(tempfile.mkdtemp(prefix="rag_entail_")) / "chroma", collection_name="real")

    for pdf in sorted(corpus.glob("*.pdf")):
        raw = truncate(pdf, PAGE_CAP)
        result = ingest_pdf(f"{pdf.stem}.pdf", raw, store=store)
        print(f"indexed {pdf.stem}.pdf: {result['chunks']} chunks, ocr={result['ocr_pages']}")
    return store, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Out-of-sample entailment check on real-corpus claims")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval" / "real_corpus")
    parser.add_argument("--report", type=Path, default=ROOT / "eval" / "reports" / "real_corpus.json")
    parser.add_argument("--threshold", type=float, default=config.ENTAIL_THRESHOLD)
    parser.add_argument("--k", type=int, default=config.CONTEXT_K)
    parser.add_argument("--retrieve-k", type=int, default=config.RETRIEVE_K)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "rag_real_entail_index",
        help="cached Chroma directory; pass an empty value to force a fresh index",
    )
    parser.add_argument("--dump", type=Path, default=ROOT / "eval" / "reports" / "entail_real.json")
    parser.add_argument("--snippet", type=int, default=220)
    args = parser.parse_args()

    stored = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [row for row in stored["rows"] if row.get("claims")]
    if args.limit:
        rows = rows[: args.limit]

    store, reused = build_store(args.corpus, args.index_dir)
    print(f"index {'reused from cache' if reused else 'built fresh'}; scoring {len(rows)} answers at threshold {args.threshold}")

    retriever = Retriever(store)
    results: list[dict] = []
    agreement = matched = 0
    for row in rows:
        contexts = retriever.retrieve(
            row["question"],
            k=args.k,
            retrieve_k=args.retrieve_k,
            cross_encoder=True,
        )
        replayed = [(chunk.source, chunk.page) for chunk in contexts] == [
            (item["source"] if isinstance(item, dict) else item[0], item["page"] if isinstance(item, dict) else item[1])
            for item in row["retrieved"]
        ]
        matched += 1 if replayed else 0
        claims = []
        skipped_meta = 0
        for claim in row["claims"]:
            if is_meta_claim(claim["sentence"]):
                skipped_meta += 1
                continue
            blocks = [n for n in claim.get("cited_blocks") or [] if 1 <= n <= len(contexts)]
            if not blocks:
                continue
            premise = "\n".join(contexts[n - 1].text for n in blocks)
            score = entail_max(premise, claim["sentence"])
            supported = score >= args.threshold
            agreement += 1 if supported == (claim.get("verdict") == "supported") else 0
            claims.append(
                {
                    "sentence": claim["sentence"],
                    "cited_blocks": blocks,
                    "lexical_verdict": claim.get("verdict"),
                    "lexical_overlap": claim.get("cited_overlap"),
                    "entail_score": score,
                    "entail_supported": supported,
                    "block_page": [contexts[n - 1].page for n in blocks],
                    "block_snippet": premises_snippet(premise, args.snippet),
                }
            )
        results.append(
            {"id": row["id"], "doc": row["doc"], "replayed": replayed, "skipped_meta": skipped_meta, "claims": claims}
        )

    total = sum(len(item["claims"]) for item in results)
    skipped = sum(item["skipped_meta"] for item in results)
    flagged = [claim for item in results for claim in item["claims"] if not claim["entail_supported"]]
    summary = {
        "answers": len(results),
        "claims": total,
        "skipped_meta_claims": skipped,
        "nli_supported": total - len(flagged),
        "nli_flagged": len(flagged),
        "agrees_with_lexical": agreement,
        "replayed_rows": matched,
        "threshold": args.threshold,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.ENTAIL_MODEL,
        "threshold": args.threshold,
        "summary": summary,
        "rows": results,
    }
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    args.dump.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"\nclaims: {total} (plus {skipped} meta sentences skipped) | supported by NLI: {total - len(flagged)} "
        f"| flagged: {len(flagged)} | agrees with lexical: {agreement}/{total} | replay matched: {matched}/{len(results)}"
    )
    for claim in flagged:
        print(f"\n[{claim['entail_score']:.3f}] {claim['sentence']}")
        print(f"    cited {claim['cited_blocks']} on pages {claim['block_page']} | lexical {claim['lexical_verdict']} {claim['lexical_overlap']}")
        print(f"    block: {claim['block_snippet']}")
    print(f"\nreport: {args.dump}")


def premises_snippet(text: str, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


if __name__ == "__main__":
    main()
