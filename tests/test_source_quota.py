"""Per-document quota: the same candidate budget, spent across documents.

The measured failure this fixes: on eight arXiv papers a two-document question got
four context blocks from one PDF at `k=4` and seven of eight at `k=8`, because a
global top-k query is answered by whichever document resembles the question most.
A quota asks each document for its best few instead, so the second paper is in the
pool whether or not it outranks the first.
"""

from __future__ import annotations

from src.store import RetrievedChunk


def chunk(source: str, page: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(text=f"{source} p{page}", source=source, page=page, score=score,
                          embedding=[score, 0.0])


class OneDocumentStore:
    """A store whose global query is dominated by one document, like the real index."""

    def __init__(self):
        self.dominant = [chunk("a.pdf", page, 0.90 - page * 0.01) for page in range(1, 6)]
        self.other = {name: [chunk(name, 1, 0.55), chunk(name, 2, 0.50)] for name in ("b.pdf", "c.pdf")}

    def query(self, embedding, n_results, sources=None):
        return self.dominant[:n_results]

    def query_per_source(self, embedding, per_source, sources=None):
        pool = [c for group in (self.dominant, *self.other.values()) for c in group[:per_source]]
        pool.sort(key=lambda c: c.score, reverse=True)
        return pool


def test_global_query_never_sees_the_second_document():
    """The control case, stated as a test so the fix has something to be a fix of."""
    from src.retrieval import Retriever

    retriever = Retriever(OneDocumentStore())
    out = retriever.retrieve("q", k=4, cross_encoder=False, rerank=False)
    assert {c.source for c in out} == {"a.pdf"}


def test_quota_puts_every_document_in_the_prompt():
    import src.retrieval as retrieval

    retriever = retrieval.Retriever(OneDocumentStore())
    out = retriever.retrieve("q", k=4, cross_encoder=False, rerank=False, source_quota=2)
    sources = [c.source for c in out]
    assert set(sources) == {"a.pdf", "b.pdf", "c.pdf"}
    # the budget is unchanged: four blocks, not one per document plus a full top-4
    assert len(out) == 4
    assert retriever.last_stages["source_quota"] == 2


def test_quota_of_zero_is_the_shipped_behaviour():
    import src.retrieval as retrieval

    retriever = retrieval.Retriever(OneDocumentStore())
    out = retriever.retrieve("q", k=4, cross_encoder=False, rerank=False, source_quota=0)
    assert {c.source for c in out} == {"a.pdf"}
    assert retriever.last_stages["source_quota"] == 0


def test_quota_is_read_from_config_when_not_given(monkeypatch):
    import src.config
    import src.retrieval as retrieval

    monkeypatch.setattr(src.config, "SOURCE_QUOTA", 3)
    retriever = retrieval.Retriever(OneDocumentStore())
    out = retriever.retrieve("q", k=4, cross_encoder=False, rerank=False)
    assert {c.source for c in out} <= {"a.pdf", "b.pdf", "c.pdf"}
    assert len(out) == 4
    assert retriever.last_stages["source_quota"] == 3


def test_a_store_without_the_quota_method_still_works():
    """The quota is an optional capability: an old fake or a filtered store keeps the
    global path instead of raising. ``hasattr`` is the switch, so a test double that
    only implements ``query`` is still a valid store."""

    class LegacyStore:
        def query(self, embedding, n_results, sources=None):
            return [chunk("a.pdf", 1, 0.9)]

    import src.retrieval as retrieval

    retriever = retrieval.Retriever(LegacyStore())
    out = retriever.retrieve("q", k=2, cross_encoder=False, rerank=False, source_quota=2)
    assert len(out) == 1
    assert retriever.last_stages["source_quota"] == 2
