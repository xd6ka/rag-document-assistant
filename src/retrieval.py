"""Retrieval pipeline: vector search followed by MMR reranking.

Pull a wide candidate set from the index (``RETRIEVE_K``), then pick
``CONTEXT_K`` chunks with Maximal Marginal Relevance. Without this step a
question whose answer sits in three adjacent chunks gets three near-identical
excerpts and wastes most of the prompt budget.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import config
from .embeddings import embed_query
from .store import DocumentStore, RetrievedChunk


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def mmr_select(
    candidates: Sequence[RetrievedChunk],
    k: int,
    lambda_: float = 0.7,
) -> list[RetrievedChunk]:
    """Greedy MMR: relevance to the query, penalised by similarity to chosen chunks."""
    if not candidates or k <= 0:
        return []

    chosen = [candidates[0]]
    pool = list(candidates[1:])

    while pool and len(chosen) < k:
        best_index, best_score = 0, float("-inf")
        for index, candidate in enumerate(pool):
            relevance = candidate.score
            redundancy = max(
                (cosine_similarity(candidate.embedding, picked.embedding) for picked in chosen),
                default=0.0,
            )
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        chosen.append(pool.pop(best_index))

    return chosen


class Retriever:
    def __init__(self, store: Optional[DocumentStore] = None):
        self.store = store or DocumentStore()

    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
        retrieve_k: Optional[int] = None,
        sources: Optional[Sequence[str]] = None,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        candidates = self.store.query(
            embed_query(question),
            retrieve_k or config.RETRIEVE_K,
            sources=sources,
        )
        if not rerank:
            return candidates[: (k or config.CONTEXT_K)]
        return mmr_select(candidates, k or config.CONTEXT_K, config.MMR_LAMBDA)
