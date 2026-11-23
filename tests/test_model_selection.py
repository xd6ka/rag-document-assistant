"""Model-selection tests: the ablation must actually swap the models.

Both failures these cases guard against are silent. A one-slot embedding cache
returns the English model for a multilingual request, and a cached cross-encoder
ignores the requested checkpoint; in either case the run finishes, the report
fills with numbers, and the numbers belong to the wrong model.
"""

from __future__ import annotations

import sys
import types

import pytest

from src import embeddings, rerank
from src.store import RetrievedChunk


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    embeddings._models.clear()
    rerank.load_cross_encoder.cache_clear()
    yield
    embeddings._models.clear()
    rerank.load_cross_encoder.cache_clear()


def install_fake_sentence_transformers(monkeypatch, created: list[str], scores=None):
    """Register a stub ``sentence_transformers`` module and record what was built."""
    module = types.ModuleType("sentence_transformers")

    class FakeEncoder:
        def __init__(self, name, **kwargs):
            created.append(name)

        def encode(self, texts, **kwargs):
            # A real encode returns a numpy array, so keep the .tolist() call honest.
            import numpy as np

            return np.array([[1.0, 0.0] for _ in texts])

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            created.append(name)

        def predict(self, pairs):
            if scores is not None:
                return scores
            return [1.0 if "good" in passage else 0.0 for _, passage in pairs]

    module.SentenceTransformer = FakeEncoder
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def test_cache_is_keyed_by_model_name(monkeypatch):
    created: list[str] = []
    install_fake_sentence_transformers(monkeypatch, created)

    embeddings.embed_query("question", model_name="model-a")
    embeddings.embed_query("вопрос", model_name="model-b")
    embeddings.embed_query("question again", model_name="model-a")

    # The second call must build model-b instead of reusing model-a, and the third
    # must reuse model-a rather than build it twice.
    assert created == ["model-a", "model-b"]


def test_reranker_uses_the_requested_model(monkeypatch):
    created: list[str] = []
    install_fake_sentence_transformers(monkeypatch, created, scores=[0.0, 1.0])

    weak = RetrievedChunk(text="weak passage", source="a.pdf", page=1, score=0.90)
    strong = RetrievedChunk(text="good passage", source="b.pdf", page=2, score=0.40)

    ordered = rerank.rerank_candidates("question", [weak, strong], model_name="rerank-model")

    assert created == ["rerank-model"]
    assert [chunk.text for chunk in ordered] == ["good passage", "weak passage"]
    # Cross-encoder score is normalised into rerank_score; cosine score is untouched
    # because the UI shows it.
    assert strong.rerank_score == 1.0
    assert weak.rerank_score == 0.0
    assert strong.score == 0.40 and weak.score == 0.90
