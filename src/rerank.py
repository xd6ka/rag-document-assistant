"""Cross-encoder reranking: a second pass that reads the query and the passage together.

The vector index compares two embeddings that never saw each other, which is fast
and blunt. A cross-encoder scores each (query, passage) pair jointly, so it can
reject a chunk that merely shares vocabulary with the question. It costs one
forward pass per candidate, which is why it runs on the top ``RETRIEVE_K``
candidates rather than the whole corpus.

``model_name`` is a parameter rather than a module constant so an ablation can
compare an English reranker against a multilingual one in the same process. The
``lru_cache`` is keyed on the name for the same reason.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

from . import config
from .store import RetrievedChunk


@lru_cache(maxsize=4)
def load_cross_encoder(model_name: Optional[str] = None):
    """Lazy per-model cache: the model is only loaded when reranking is actually used."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name or config.RERANK_MODEL, device="cpu", max_length=512)


def rerank_candidates(
    question: str,
    candidates: Sequence[RetrievedChunk],
    keep: Optional[int] = None,
    model_name: Optional[str] = None,
) -> list[RetrievedChunk]:
    """Reorder candidates by cross-encoder score and return the best ``keep``.

    The original cosine ``score`` is left untouched because the UI shows it to
    users; the cross-encoder value lands in ``rerank_score``, min-max normalised
    to 0..1 so it can be mixed with cosine relevance inside MMR without one
    term swamping the other.
    """
    if not candidates:
        return []

    model = load_cross_encoder(model_name)
    pairs = [(question, candidate.text) for candidate in candidates]
    raw = [float(value) for value in model.predict(pairs)]

    lo, hi = min(raw), max(raw)
    for candidate, value in zip(candidates, raw):
        candidate.rerank_score = 0.5 if hi == lo else round((value - lo) / (hi - lo), 4)

    ordered = [candidate for _, candidate in sorted(zip(raw, candidates), key=lambda p: p[0], reverse=True)]
    return ordered[:keep] if keep else ordered
