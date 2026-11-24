"""Unit tests for the prompt budget — no model is called."""

from __future__ import annotations

from src import config
from src.generation import build_prompt, trim_block
from src.store import RetrievedChunk

BLOCK = (
    "Vector Stores for a Single-Machine Pipeline - page 2. "
    "ChromaDB builds HNSW graphs underneath. The defaults in recent versions are M=16 neighbours "
    "per node and ef_construction=200. "
    "Distance space matters more than the algorithm. Cosine space requires normalized vectors. "
    "A collection is the unit of isolation. One collection per tenant keeps filters cheap."
)
QUESTION = "What is the default number of HNSW neighbours per node in ChromaDB?"


def test_budget_zero_keeps_the_block_whole():
    assert trim_block(BLOCK, QUESTION, 0) == BLOCK
    assert trim_block(BLOCK, QUESTION, len(BLOCK) + 100) == BLOCK


def test_budget_keeps_the_relevant_sentence_and_drops_the_rest():
    trimmed = trim_block(BLOCK, QUESTION, 220)
    assert len(trimmed) <= 220
    assert "M=16" in trimmed
    assert "collection is the unit of isolation" not in trimmed


def test_kept_sentences_keep_their_original_order():
    # relevance ranking decides *which* sentences survive; the block must still read
    # as prose, so surviving sentences are re-ordered by their position.
    trimmed = trim_block(BLOCK, QUESTION, 260)
    first = trimmed.find("HNSW")
    second = trimmed.find("Cosine") if "Cosine" in trimmed else len(trimmed)
    assert first != -1
    assert first < second


def test_short_block_and_empty_question_fall_back_safely():
    short = "ChromaDB uses M=16."
    assert trim_block(short, QUESTION, 50) == short
    # no content words in the question means no ranking signal: fall back to the head
    assert len(trim_block(BLOCK, "what is this", 100)) <= 100


def test_build_prompt_applies_the_budget_to_every_block():
    contexts = [
        RetrievedChunk(text=BLOCK, source="vector-store-notes.pdf", page=2, score=0.6),
        RetrievedChunk(text=BLOCK, source="prompt-design.pdf", page=3, score=0.5),
    ]
    full = build_prompt(QUESTION, contexts, budget=0)
    trimmed = build_prompt(QUESTION, contexts, budget=200)
    assert len(trimmed) < len(full)
    assert "[1]" in trimmed and "[2]" in trimmed
    # the block text must not be preceded by its file name: a 3B model reads the
    # header as part of the sentence ("…is called source: x.pdf, page 3 …")
    assert "Context:\n[1] " in trimmed
    for number, source in ((1, "vector-store-notes.pdf"), (2, "prompt-design.pdf")):
        assert f"[{number}] {source}, page" in trimmed.split("Sources:")[1]
    assert "Sources:" in trimmed
    # the citation instruction survives trimming — it is not part of the blocks
    assert "End every sentence with the block number" in trimmed


def test_config_budget_is_an_integer_and_on_by_default():
    assert isinstance(config.CONTEXT_BUDGET, int)
    # 400 chars per block is the shipped default: the sweep in the README shows the
    # same keyword accuracy at 45 % fewer prompt characters, and one more point of
    # citation-gold accuracy.
    assert config.CONTEXT_BUDGET == 400
