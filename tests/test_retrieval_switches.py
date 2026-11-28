"""Reranker switches: the cross-encoder can be off, and its absence must be explicit.

``Retriever.last_stages`` is what the UI and the ablation read to describe a run;
if a flag were silently dropped, the report would claim a stage that never ran.
"""

from __future__ import annotations

from src.retrieval import Retriever


class FakeStore:
    """Minimal stand-in for DocumentStore: three candidates with known pages."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.queried = 0

    def query(self, embedding, n_results, sources=None):
        self.queried += 1
        return list(self._chunks[:n_results])


def make_chunks():
    from src.store import RetrievedChunk

    return [
        RetrievedChunk(text=f"chunk {i}", source="a.pdf", page=i + 1, score=1.0 - i * 0.1, embedding=[1.0, float(i)])
        for i in range(3)
    ]


def test_cross_encoder_off_is_recorded_and_skips_rerank(monkeypatch):
    import src.retrieval as retrieval

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("cross-encoder was called while disabled")

    monkeypatch.setattr(retrieval, "rerank_candidates", explode)
    retriever = Retriever(FakeStore(make_chunks()))
    out = retriever.retrieve("q", k=2, cross_encoder=False)
    assert retriever.last_stages == {"cross_encoder": False, "mmr": True, "source_quota": 0}
    assert len(out) == 2


def test_cross_encoder_on_is_recorded(monkeypatch):
    import src.retrieval as retrieval

    def fake_rerank(question, candidates, model_name=None):
        # model_name is part of the real signature: the ablation picks the checkpoint
        # per run, so retrieval always passes it through.
        for index, chunk in enumerate(candidates):
            chunk.rerank_score = 1.0 - index * 0.1
        return candidates

    monkeypatch.setattr(retrieval, "rerank_candidates", fake_rerank)
    retriever = Retriever(FakeStore(make_chunks()))
    out = retriever.retrieve("q", k=2, cross_encoder=True, rerank=False)
    assert retriever.last_stages == {"cross_encoder": True, "mmr": False, "source_quota": 0}
    assert out[0].rerank_score is not None


def test_mmr_off_returns_top_cosine_order():
    retriever = Retriever(FakeStore(make_chunks()))
    out = retriever.retrieve("q", k=2, cross_encoder=False, rerank=False)
    assert [chunk.page for chunk in out] == [1, 2]
