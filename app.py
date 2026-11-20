"""Index PDFs from the command line.

    python app.py                      # every PDF in data/
    python app.py data/real.pdf        # one specific file
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.ingest import ingest_path
from src.store import DocumentStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF files into the vector store")
    parser.add_argument(
        "paths",
        nargs="*",
        help="PDF files to index; defaults to every PDF inside data/",
    )
    args = parser.parse_args()

    targets = [Path(p) for p in args.paths]
    if not targets:
        targets = sorted((config.PROJECT_ROOT / "data").glob("*.pdf"))

    if not targets:
        print("No PDFs found. Put a file in data/ or pass a path explicitly.")
        return

    store = DocumentStore()
    for path in targets:
        if not path.exists():
            print(f"skip {path}: file not found")
            continue
        try:
            result = ingest_path(path, store=store)
        except Exception as exc:  # noqa: BLE001 - report per file and continue
            print(f"fail {path.name}: {exc}")
            continue

        if result["skipped"]:
            print(f"skip {path.name}: no extractable text")
        else:
            print(f"ok   {path.name}: {result['chunks']} chunks from {result['pages']} pages")

    print(f"\nCollection now holds {store.count()} chunks.")


if __name__ == "__main__":
    main()
