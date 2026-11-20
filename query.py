"""Interactive CLI: ask questions about indexed documents and print citations."""

from __future__ import annotations

import argparse

from src import config
from src.generation import answer_question
from src.retrieval import Retriever
from src.store import DocumentStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about indexed PDFs")
    parser.add_argument("--k", type=int, default=config.CONTEXT_K, help="chunks in context")
    parser.add_argument("--source", nargs="*", default=None, help="restrict to these documents")
    parser.add_argument("--no-mmr", action="store_true", help="disable MMR reranking")
    args = parser.parse_args()

    store = DocumentStore()
    if store.count() == 0:
        print("The collection is empty. Index a PDF first: python app.py data/real.pdf")
        return

    retriever = Retriever(store)
    print("RAG Assistant ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        contexts = retriever.retrieve(
            question,
            k=args.k,
            sources=args.source or None,
            rerank=not args.no_mmr,
        )
        if not contexts:
            print("Nothing relevant found in the indexed documents.")
            continue

        try:
            print(f"\nAI: {answer_question(question, contexts)}")
        except Exception as exc:  # noqa: BLE001 - CLI should not traceback on API errors
            print(f"\nAI: API error: {exc}")

        print("\nSources:")
        for number, chunk in enumerate(contexts, start=1):
            print(f"  [{number}] {chunk.source} p.{chunk.page} (similarity {chunk.score:.3f})")


if __name__ == "__main__":
    main()
