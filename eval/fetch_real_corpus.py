"""Download the real-world PDFs used by eval/run_real.py.

The synthetic corpus in eval/corpus/ was written by me, so it is suspiciously
friendly: two columns of prose, one idea per page, no tables or footnotes.
This script pulls six genuine documents with layouts I did not design, so the
retrieval numbers can be checked against something adversarial.

PDFs are not committed (licences, size). Run this once, then eval/run_real.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "real_corpus"
MANIFEST = HERE / "real_corpus_manifest.json"

# name -> (url, why this one is in the set)
DOCS = {
    "attention": (
        "https://arxiv.org/pdf/1706.03762v7",
        "single column, tables and footnotes, ~15 pages",
    ),
    "rag": (
        "https://arxiv.org/pdf/2005.11401v4",
        "two column, dense related-work prose",
    ),
    "bert": (
        "https://arxiv.org/pdf/1810.04805v2",
        "two column, many tables and ablation sections",
    ),
    "nist-800-63b": (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf",
        "formal standard: numbered clauses, running headers and footers",
    ),
    "gpo-911": (
        "https://www.govinfo.gov/content/pkg/GPO-911REPORT/pdf/GPO-911REPORT.pdf",
        "government commission report: 585 pages, photos with captions, footnotes",
    ),
    "bills-hr3684": (
        "https://www.govinfo.gov/content/pkg/BILLS-117hr3684enr/pdf/BILLS-117hr3684enr.pdf",
        "legislative text: numbered sections, dense cross-references, no prose",
    ),
    # Second round: layouts the first six did not cover. A dead rfc6750 entry was
    # dropped from this list — rfc-editor.org 500s on its PDF endpoint and
    # datatracker returns 403, so the IETF-memo case stays unmeasured rather than
    # quietly failing in the manifest.
    "patent-us9600411": (
        "https://patentimages.storage.googleapis.com/28/71/84/848d5050ce2b50/US9600411.pdf",
        "patent: claims, numbered paragraphs, figure references, two-column OCR-like text",
    ),
    # fiscal.treasury.gov/files/.../FR-2023.pdf answers with a 119 KB HTML page
    # (a PDF reader reports "invalid pdf header: b'<!DOC'"), so the financial-report
    # slot is filled by the Federal Reserve's annual report instead.
    "fed-annual-2023": (
        "https://www.federalreserve.gov/publications/files/2023-annual-report.pdf",
        "central-bank annual report: financial statements, tables split across pages, footnotes",
    ),
    "ipcc-ar6-ch3": (
        "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter03.pdf",
        "scientific assessment chapter: dense figures, captions, cross-chapter references",
    ),
    "dropout-2012": (
        "https://arxiv.org/pdf/1207.0580v1",
        "older preprint: single column, heavy notation, experiment tables",
    ),
}

USER_AGENT = "Mozilla/5.0 (compatible; rag-document-assistant eval; +local)"


def fetch(name: str, url: str, dest: Path) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    return {
        "file": dest.name,
        "url": url,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for name, (url, note) in DOCS.items():
        dest = OUT_DIR / f"{name}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            report[name] = {
                "file": dest.name,
                "url": url,
                "bytes": dest.stat().st_size,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "note": note,
                "cached": True,
            }
            continue
        try:
            info = fetch(name, url, dest)
            info["note"] = note
            info["cached"] = False
            report[name] = info
            print(f"ok   {name:14s} {info['bytes']:>9,d} bytes  {note}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"fail {name:14s} {exc}")
            report[name] = {"url": url, "note": note, "error": str(exc)}

    MANIFEST.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = [k for k, v in report.items() if "error" in v]
    print(f"\n{len(report) - len(failures)}/{len(report)} downloaded -> {MANIFEST}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
