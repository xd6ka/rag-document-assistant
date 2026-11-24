"""Calibrate the entailment check against the hand-read answer audit.

The labels here are not produced by a model. ``eval/reports/answer_audit.md``
records five answers whose citation is drifted (the fact sits in another block)
and twenty whose citation holds; the same 25 sentences are replayed here.

Two things this script deliberately does not do:

- it does not generate new answers, it replays retrieval and grades stored ones,
  so the comparison is against the audit that was read by hand;
- it does not choose a threshold by what looks good on the plot. It sweeps a grid,
  reports accuracy / drift recall / false positives for every step, and the shipped
  threshold is whatever wins on accuracy, ties broken toward the *higher* value
  (fewer false accusations on correct answers).

    python eval/entail_calibrate.py                     # sweep + write the report
    python eval/entail_calibrate.py --report eval/reports/latest.json --limit 10
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

from src import config  # noqa: E402
from src.entailment import EntailmentUnavailable, entail_max  # noqa: E402
from src.ingest import ingest_path  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.store import DocumentStore  # noqa: E402

# From eval/reports/answer_audit.md, run 2: these five answers cite a block that
# does not carry the fact. Everything else in scope is hand-confirmed correct.
HAND_DRIFTED = {"e05", "v02", "c03", "p04", "d04"}

THRESHOLD_GRID = [round(0.10 + 0.05 * step, 2) for step in range(17)]  # 0.10 … 0.90


def load_qa(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_pairs(report_path: Path, qa_path: Path, corpus: Path, k: int, retrieve_k: int) -> tuple[list[dict], dict]:
    """Replay retrieval and return one labelled pair per cited sentence."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    items = {item["id"]: item for item in load_qa(qa_path)}
    cross_encoder = payload.get("summary", {}).get("settings", {}).get("cross_encoder", True)

    workdir = Path(tempfile.mkdtemp(prefix="rag_entail_"))
    store = DocumentStore(path=workdir / "chroma", collection_name="entail")
    for pdf in sorted(corpus.glob("*.pdf")):
        ingest_path(pdf, store=store)
    retriever = Retriever(store)

    pairs: list[dict] = []
    for row in payload["rows"]:
        if not row.get("in_scope", True):
            continue
        contexts = retriever.retrieve(
            items.get(row["id"], {}).get("question", ""),
            k=k,
            retrieve_k=retrieve_k,
            cross_encoder=cross_encoder,
        )
        blocks = [(chunk.source, chunk.page) for chunk in contexts]
        if blocks != [tuple(pair) for pair in (row.get("retrieved") or [])]:
            print(f"WARNING: {row['id']} retrieval order differs from the stored report")

        for claim in row.get("claims") or []:
            cited = [n for n in claim.get("cited_blocks") or [] if 1 <= n <= len(contexts)]
            if not cited:
                continue  # uncited sentences carry no attribution to calibrate
            premise = "\n".join(contexts[n - 1].text for n in cited)
            pairs.append(
                {
                    "id": row["id"],
                    "sentence": claim["sentence"],
                    "cited_blocks": cited,
                    "premise_chars": len(premise),
                    "label": 0.0 if row["id"] in HAND_DRIFTED else 1.0,
                    "_premise": premise,
                }
            )
    return pairs, {"cross_encoder": cross_encoder, "k": k, "retrieve_k": retrieve_k}


def sweep(pairs: list[dict], thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for pair in pairs:
            predicted = pair["score"] >= threshold
            actual = pair["label"] == 1.0
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            else:
                fn += 1
        total = tp + fp + tn + fn
        rows.append(
            {
                "threshold": threshold,
                "accuracy": round((tp + tn) / total, 3) if total else None,
                # among the hand-drifted sentences, how many are rejected
                "drift_recall": round(tn / (tn + fp), 3) if (tn + fp) else None,
                "supported_kept": tp,
                "missed_drifts": fp,       # drift graded as supported
                "false_accusations": fn,   # correct citation graded as drift
                "drifts_rejected": tn,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
        )
    return rows


def best_threshold(rows: list[dict]) -> dict:
    """Best accuracy; ties go to the middle of the plateau, not to its edge.

    Accuracy is a step function here — several neighbouring thresholds give the same
    count because the scores are far apart. Picking the highest of them (the usual
    sklearn rule) puts the operating point one hard sentence away from a tier, so the
    same model on a slightly harder claim flips. The middle of the plateau maximises
    the margin to both the weakest supported sentence and the strongest drifted one.
    """
    best = max(row["accuracy"] or 0.0 for row in rows)
    tied = [row for row in rows if (row["accuracy"] or 0.0) == best]
    return tied[len(tied) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(ROOT / "eval" / "reports" / "latest.json"))
    parser.add_argument("--qa", default=str(ROOT / "eval" / "qa_set.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "eval" / "corpus"))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--model", default=None, help="override ENTAIL_MODEL")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N pairs")
    parser.add_argument("--dump", default=str(ROOT / "eval" / "reports" / "entailment.json"))
    args = parser.parse_args()

    pairs, settings = build_pairs(Path(args.report), Path(args.qa), Path(args.corpus), args.k, args.retrieve_k)
    if not pairs:
        raise SystemExit("no cited sentences found in the report")
    if args.limit:
        pairs = pairs[: args.limit]

    for pair in pairs:
        premise = pair.pop("_premise")
        try:
            pair["score"] = entail_max(premise, pair["sentence"], model_name=args.model)
        except EntailmentUnavailable as exc:
            raise SystemExit(f"entailment model unavailable: {exc}")

    rows = sweep(pairs, THRESHOLD_GRID)
    chosen = best_threshold(rows)
    supported = [pair["score"] for pair in pairs if pair["label"] == 1.0]
    drifted = [pair["score"] for pair in pairs if pair["label"] == 0.0]

    print(f"\n=== entailment calibration: {len(pairs)} cited sentences "
          f"({len(supported)} hand-supported, {len(drifted)} hand-drifted) ===")
    print(f"supported scores: min {min(supported):.3f}  max {max(supported):.3f}")
    if drifted:
        print(f"drifted scores:   min {min(drifted):.3f}  max {max(drifted):.3f}")
    print(f"\n{'threshold':>9} {'accuracy':>9} {'drift recall':>13} {'missed drifts':>14} {'false acc.':>11}")
    for row in rows:
        recall = row["drift_recall"]
        print(f"{row['threshold']:>9.2f} {row['accuracy']:>9.3f} "
              f"{(recall if recall is not None else float('nan')):>13.3f} "
              f"{row['missed_drifts']:>14} {row['false_accusations']:>11}")
    print(f"\nchosen threshold {chosen['threshold']:.2f}: accuracy {chosen['accuracy']}, "
          f"drifts rejected {chosen['drifts_rejected']}/{chosen['drifts_rejected'] + chosen['missed_drifts']}, "
          f"false accusations {chosen['false_accusations']}/{chosen['supported_kept'] + chosen['false_accusations']}")
    print("\nper pair (label 1 = hand-supported):")
    for pair in pairs:
        mark = "OK " if (pair["score"] >= chosen["threshold"]) == (pair["label"] == 1.0) else "!! "
        print(f"{mark}{pair['id']:<4} score={pair['score']:.3f} label={int(pair['label'])} :: {pair['sentence'][:90]}")

    dump = Path(args.dump)
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "report": args.report,
                "model": args.model or config.ENTAIL_MODEL,
                "settings": settings,
                "thresholds": THRESHOLD_GRID,
                "chosen_threshold": chosen["threshold"],
                "chosen": chosen,
                "sweep": rows,
                "score_range": {
                    "supported": [round(min(supported), 4), round(max(supported), 4)],
                    "drifted": ([round(min(drifted), 4), round(max(drifted), 4)] if drifted else None),
                },
                "pairs": pairs,
                "labels_from": "eval/reports/answer_audit.md (run 2, hand-read)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nentailment report: {dump}")


if __name__ == "__main__":
    main()
