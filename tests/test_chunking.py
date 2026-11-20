"""Unit tests for page-aware chunking.

Run with: pytest -q
"""

from __future__ import annotations

import pytest

from src.chunking import _windows, chunk_page, chunk_pages

REPEATED = "Sentence one about fonts and subsets. " * 60  # ~2300 characters


def test_chunks_never_exceed_size():
    for chunk in chunk_page(REPEATED, page=1, size=400, overlap=80):
        assert len(chunk.text) <= 400


def test_consecutive_chunks_share_overlap():
    chunks = chunk_page(REPEATED, page=1, size=400, overlap=80)
    assert len(chunks) >= 3
    for previous, following in zip(chunks, chunks[1:]):
        assert following.text[:30] in previous.text


def test_chunks_keep_exact_page_numbers():
    chunks = chunk_pages(["alpha " * 200, "", "beta " * 200], size=300, overlap=50)
    assert {chunk.page for chunk in chunks} == {1, 3}


def test_short_tail_is_merged_into_previous_chunk():
    chunks = chunk_page("a" * 500 + " tail", page=1, size=500, overlap=0)
    assert len(chunks) == 1


def test_empty_page_produces_no_chunks():
    assert chunk_page("", page=1, size=300, overlap=50) == []


def test_chunk_indices_are_sequential():
    chunks = chunk_page(REPEATED, page=2, size=300, overlap=50)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        list(_windows("abc", size=10, overlap=10))

    with pytest.raises(ValueError):
        list(_windows("abc", size=10, overlap=-1))


def test_window_cuts_on_whitespace_not_mid_word():
    text = "word " * 100
    chunks = chunk_page(text, page=1, size=120, overlap=20)
    for chunk in chunks:
        assert not chunk.text.endswith("wor")
