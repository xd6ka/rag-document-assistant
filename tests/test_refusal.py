"""Tests for refusal detection.

The first version matched a fixed marker list and scored a real refusal as a
failure, because the model phrased it as "does not provide information in the
given context". Every case below is a phrasing the corpus run actually produced
or is one edit away from it.
"""

from __future__ import annotations

from eval.run_eval import is_refusal


def test_marker_phrasing_is_a_refusal():
    assert is_refusal("I do not know [1].") is True


def test_negation_pattern_catches_does_not_provide():
    assert is_refusal("[1] DashScope does not provide information in the given context. [1]") is True


def test_does_not_contain_information_is_a_refusal():
    assert is_refusal("[1] The given text does not contain information about the winner. [1]") is True


def test_russian_refusal_is_recognised():
    assert is_refusal("В документе нет информации об этом, не знаю.") is True


def test_a_real_answer_is_not_a_refusal():
    answer = "all-MiniLM-L6-v2 has 22 million parameters [1]."
    assert is_refusal(answer) is False


def test_an_answer_mentioning_that_context_contains_the_fact_is_not_a_refusal():
    answer = "The context states that cosine space requires normalized vectors [1]."
    assert is_refusal(answer) is False
