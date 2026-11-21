"""Retrieval-stage tests: ranking by rerank score, and citation-marker mapping.

The cross-encoder model itself is not loaded here — these cases pin the plumbing
that decides which score wins, which is where a silent regression would live.
"""

from __future__ import annotations

from eval.run_eval import cited_blocks, parse_citations
from src.retrieval import mmr_select
from src.store import RetrievedChunk


def make(text: str, score: float, embedding: list[float], rerank_score=None) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source="doc.pdf",
        page=1,
        score=score,
        embedding=embedding,
        rerank_score=rerank_score,
    )


def test_relevance_falls_back_to_cosine_score():
    chunk = make("a", 0.8, [1.0, 0.0])
    assert chunk.relevance == 0.8


def test_relevance_prefers_rerank_score():
    chunk = make("a", 0.8, [1.0, 0.0], rerank_score=0.1)
    assert chunk.relevance == 0.1


def test_mmr_follows_rerank_order_not_cosine_order():
    # Cosine order puts the first chunk on top; the cross-encoder disagrees.
    strong_cosine = make("cosine-favourite", 0.90, [1.0, 0.0], rerank_score=0.0)
    strong_rerank = make("rerank-favourite", 0.60, [0.0, 1.0], rerank_score=1.0)
    chosen = mmr_select([strong_cosine, strong_rerank], k=2, lambda_=0.7)
    assert [chunk.text for chunk in chosen] == ["cosine-favourite", "rerank-favourite"]

    # With relevance ranked by the cross-encoder, the order flips.
    chosen = mmr_select([strong_rerank, strong_cosine], k=2, lambda_=0.7)
    assert [chunk.text for chunk in chosen] == ["rerank-favourite", "cosine-favourite"]


def test_parse_citations_reads_every_marker():
    assert parse_citations("Claim one [1] and two [3], again [1].") == [1, 3, 1]


def test_cited_blocks_maps_markers_to_prompt_blocks():
    answer = "Fact [1] and detail [2]."
    retrieved = [("a.pdf", 3), ("b.pdf", 7)]
    assert cited_blocks(answer, retrieved) == [("a.pdf", 3), ("b.pdf", 7)]


def test_cited_blocks_ignores_out_of_range_markers():
    # A model can invent [9] when only four blocks were shown; that must not crash
    # or silently point at a page that was never in the prompt.
    answer = "Invented citation [9]."
    retrieved = [("a.pdf", 3)]
    assert cited_blocks(answer, retrieved) == []
