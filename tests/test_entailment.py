"""Unit tests for the entailment layer — no checkpoint is loaded."""

from __future__ import annotations

import pytest

from src import config
from src.entailment import (
    EntailmentUnavailable,
    _entail_index,
    entail_max,
    is_meta_claim,
    premise_windows,
    split_premise,
    supports,
)


class FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class FakeModel:
    def __init__(self, id2label):
        self.config = FakeConfig(id2label)


def test_split_premise_breaks_on_sentence_ends():
    text = "First claim about 768 dimensions. Second claim about 16 neighbours!\nThird question goes here?"
    parts = split_premise(text)
    assert len(parts) == 3
    assert parts[0].startswith("First claim")
    assert parts[-1].startswith("Third question")


def test_split_premise_drops_fragments_but_keeps_a_fallback():
    text = "(a)\n(b)\n"  # OCR leftovers shorter than the minimum
    assert split_premise(text) == ["(a) (b)"] or split_premise(text) == ["(a)\n(b)"]
    assert split_premise("") == []
    assert split_premise("tiny") == ["tiny"]


def test_split_premise_joins_hard_wrapped_lines():
    # the regression that made calibration look broken: chunk text is wrapped at
    # ~60 characters, and splitting on "\n" cut the number away from its subject.
    text = (
        "The models are not equal on the MTEB retrieval average:\n"
        "all-MiniLM-L6-v2 reaches 56.0, e5-small-v2\n"
        "reaches 59.8 and bge-small-en-v1.5 reaches 62.2.\n"
        "Memory matters more than people expect."
    )
    parts = split_premise(text)
    assert len(parts) == 2
    assert "bge-small-en-v1.5 reaches 62.2" in parts[0]
    assert parts[0].startswith("The models are not equal")
    assert parts[1] == "Memory matters more than people expect."


def test_split_premise_breaks_long_sentences_on_semicolons():
    text = "The settings are " + "a" * 300 + "; " + "b" * 100 + "; " + "c" * 100
    parts = split_premise(text)
    assert len(parts) == 3
    assert all(len(part) >= 15 for part in parts)


def test_premise_windows_give_each_sentence_its_neighbour():
    # "it" and "the defaults" only make sense with the sentence before them, so each
    # sentence must also travel with its neighbour.
    text = (
        "hit@k counts a question as solved if any gold item is in the top k. "
        "It is the most informative metric, because it hides how far down the answer was. "
        "MRR averages 1/rank of the first correct item."
    )
    sentences = split_premise(text)
    windows = premise_windows(text)
    assert len(sentences) == 3
    assert len(windows) == 5  # three sentences plus two overlapping pairs
    assert "hit@k counts a question as solved" in windows[3]
    assert "It is the most informative metric" in windows[3]
    assert premise_windows(text, window=1) == sentences


def test_meta_claims_are_recognised_and_skipped():
    # a sentence about the answer's coverage cannot be entailed by any block; scoring
    # it produces an accusation the audit has to throw away.
    assert is_meta_claim("The title of the 2012 preprint is not mentioned in the provided context.")
    assert is_meta_claim("There is no information about this in the context.")
    assert not is_meta_claim("The encoder is composed of a stack of N = 6 identical layers.")


def test_config_threshold_is_valid_and_off_by_default():
    assert 0.0 <= config.ENTAIL_THRESHOLD <= 1.0
    assert config.ENTAIL_ENABLED is False


def test_entail_index_reads_the_label_order_from_the_model():
    assert _entail_index(FakeModel({0: "contradiction", 1: "entailment", 2: "neutral"})) == 1
    assert _entail_index(FakeModel({0: "entailment", 1: "neutral", 2: "contradiction"})) == 0
    assert _entail_index(FakeModel({})) == 1  # documented fallback


def test_supports_uses_the_threshold_and_the_injected_scorer():
    assert supports("premise", "claim", threshold=0.5, scorer=lambda p, h: 0.9) is True
    assert supports("premise", "claim", threshold=0.5, scorer=lambda p, h: 0.2) is False
    # a score exactly on the threshold counts as support, not as a drift
    assert supports("premise", "claim", threshold=0.5, scorer=lambda p, h: 0.5) is True


def test_supports_falls_back_to_the_configured_threshold(monkeypatch):
    monkeypatch.setattr(config, "ENTAIL_THRESHOLD", 0.8)
    assert supports("premise", "claim", scorer=lambda p, h: 0.7) is False
    assert supports("premise", "claim", scorer=lambda p, h: 0.9) is True


def test_entail_max_returns_zero_without_loading_a_model():
    assert entail_max("", "claim") == 0.0
    assert entail_max("a premise sentence here", "   ") == 0.0


def test_unavailable_error_is_a_runtime_error():
    assert issubclass(EntailmentUnavailable, RuntimeError)
