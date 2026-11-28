"""Retrieval pipeline: vector search, optional cross-encoder reranking, then MMR.

Three stages, each switchable so their contribution can be measured separately:

1. ``store.query`` pulls ``RETRIEVE_K`` candidates by embedding similarity — cheap,
   but both sides of the comparison are embedder vectors that never saw each other;
2. a cross-encoder reorders those candidates by reading (query, passage) jointly;
3. MMR picks the final ``CONTEXT_K`` with redundancy taken into account, so a
   question whose answer sits in three adjacent chunks does not spend the whole
   prompt budget on three near-identical excerpts.

MMR's relevance term uses the cross-encoder score when reranking ran, because
after stage 2 the cosine order is no longer the best signal available.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import config
from .embeddings import embed_query
from .rerank import rerank_candidates
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
            relevance = candidate.relevance
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
        self.last_stages: dict[str, bool] = {}

    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
        retrieve_k: Optional[int] = None,
        sources: Optional[Sequence[str]] = None,
        rerank: bool = True,
        cross_encoder: Optional[bool] = None,
        embed_model: Optional[str] = None,
        rerank_model: Optional[str] = None,
        source_quota: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """``rerank`` toggles MMR; ``cross_encoder`` toggles stage 2 (None = config).

        ``embed_model`` / ``rerank_model`` override the configured checkpoints; the
        embedding one must match the model the store was indexed with, so an
        ablation that swaps embedders has to rebuild the collection too.

        ``source_quota`` fetches the best few chunks of *each* document instead of the
        global top-k (None = ``config.SOURCE_QUOTA``, 0 = off). The candidate budget
        is unchanged: with 8 documents and a quota of 3, that is 24 candidates, just
        spread over every paper rather than over whichever one the question resembles.
        """
        use_cross_encoder = config.RERANK_ENABLED if cross_encoder is None else cross_encoder
        quota = config.SOURCE_QUOTA if source_quota is None else source_quota
        embedding = embed_query(question, model_name=embed_model)
        if quota > 0 and hasattr(self.store, "query_per_source"):
            candidates = self.store.query_per_source(embedding, quota, sources=sources)
        else:
            candidates = self.store.query(embedding, retrieve_k or config.RETRIEVE_K, sources=sources)
        self.last_stages = {"cross_encoder": use_cross_encoder, "mmr": rerank,
                            "source_quota": quota}

        if use_cross_encoder:
            candidates = rerank_candidates(question, candidates, model_name=rerank_model)

        final_k = k or config.CONTEXT_K
        if not rerank:
            return candidates[:final_k]
        return mmr_select(candidates, final_k, config.MMR_LAMBDA)
