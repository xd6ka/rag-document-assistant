"""Tests for the gold-label validator.

The validator is what keeps the evaluation honest: if a question points at the
wrong page, or the page no longer contains the fact, the run must fail loudly
instead of reporting a number that looks good.
"""

from __future__ import annotations

from eval.run_eval import validate_qa_set


def test_accepts_correct_label():
    items = [{"id": "q1", "doc": "doc", "gold_pages": [2], "page_contains": "needle", "in_scope": True}]
    pages = {"doc.pdf": ["hay", "a needle in the text", "more hay"]}
    assert validate_qa_set(items, pages) == []


def test_flags_token_missing_from_gold_page():
    items = [{"id": "q1", "doc": "doc", "gold_pages": [1], "page_contains": "needle", "in_scope": True}]
    pages = {"doc.pdf": ["hay", "a needle in the text"]}
    problems = validate_qa_set(items, pages)
    assert any("not found" in problem for problem in problems)


def test_flags_page_out_of_range():
    items = [{"id": "q1", "doc": "doc", "gold_pages": [7], "page_contains": None, "in_scope": True}]
    pages = {"doc.pdf": ["only one page"]}
    problems = validate_qa_set(items, pages)
    assert any("does not exist" in problem for problem in problems)


def test_flags_unknown_document():
    items = [{"id": "q1", "doc": "ghost", "gold_pages": [1], "in_scope": True}]
    pages = {"doc.pdf": ["text"]}
    problems = validate_qa_set(items, pages)
    assert any("unknown document" in problem for problem in problems)


def test_out_of_scope_items_are_not_validated():
    items = [{"id": "o1", "doc": None, "gold_pages": [], "in_scope": False}]
    assert validate_qa_set(items, {"doc.pdf": ["text"]}) == []
