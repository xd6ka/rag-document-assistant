"""Streaming tests: the token order the UI renders, and no double generation.

The failure this guards against is not a crash — it is a stream that arrives out
of order, or a retry that appends a second answer behind the first, both of which
look like a model problem from the outside.
"""

from __future__ import annotations

from types import SimpleNamespace

from src import generation
from src.store import RetrievedChunk


class FakeClient:
    def __init__(self, deltas: list[str]):
        self.calls: list[dict] = []
        self.deltas = deltas
        client = self

        class Completions:
            def create(self, **kwargs):
                client.calls.append(kwargs)
                events = []
                for text in client.deltas:
                    content = text if text else None  # empty deltas are heartbeat events
                    events.append(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]))
                return events

        self.chat = SimpleNamespace(completions=Completions())


def contexts():
    return [
        RetrievedChunk(text="all-MiniLM-L6-v2 has 22 million parameters.", source="a.pdf", page=1, score=0.9),
        RetrievedChunk(text="Memory use is about 90 megabytes.", source="a.pdf", page=2, score=0.8),
    ]


def test_stream_yields_tokens_in_order_and_skips_empty_deltas(monkeypatch):
    fake = FakeClient(["all-MiniLM", "-L6", "", "-v2 has", " 22 million", " parameters [1]."])
    monkeypatch.setattr(generation, "get_client", lambda: fake)

    answer = "".join(generation.stream_answer("How many parameters?", contexts()))

    assert answer == "all-MiniLM-L6-v2 has 22 million parameters [1]."
    assert fake.calls and fake.calls[0]["stream"] is True


def test_stream_sends_the_numbered_context(monkeypatch):
    fake = FakeClient(["ok [1]"])
    monkeypatch.setattr(generation, "get_client", lambda: fake)

    list(generation.stream_answer("Question?", contexts()))

    user_message = fake.calls[0]["messages"][-1]["content"]
    # numbering is what the citation instruction refers to; the file/page mapping
    # travels in the Sources list so it cannot be read as part of a sentence
    assert "[1] all-MiniLM-L6-v2 has 22 million parameters." in user_message
    assert "Sources:" in user_message
    assert "[1] a.pdf, page 1" in user_message
    assert "[2] a.pdf, page 2" in user_message


def test_stream_does_not_retry_and_duplicate_the_answer(monkeypatch):
    fake = FakeClient(["first half", " second half"])
    monkeypatch.setattr(generation, "get_client", lambda: fake)

    list(generation.stream_answer("Question?", contexts()))

    # One request per question: a retry after tokens were emitted would append a
    # second answer behind the first.
    assert len(fake.calls) == 1
