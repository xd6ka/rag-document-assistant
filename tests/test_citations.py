"""Claim-level citation checks: parsing and verdicts.

The model is not loaded here; sentence vectors are supplied directly so the
thresholds can be exercised without a 30-second embedding warm-up.
"""

from __future__ import annotations

from src.citations import check_sentence, cited_sentences, lexical_support, support_rate


def test_trailing_marker_attaches_to_previous_sentence():
    # The pattern a 3B model actually produced: bracket after the final period.
    answer = "A link is a clickable rectangle with an action. It jumps to another page. [2]"
    sentences = cited_sentences(answer)
    assert sentences[-1][1] == [2]
    assert "jumps to another page" in sentences[-1][0]


def test_marker_only_fragment_does_not_become_a_claim():
    answer = "Cosine space requires normalized vectors. [1]"
    sentences = cited_sentences(answer)
    assert len(sentences) == 1
    assert sentences[0][1] == [1]


def test_short_filler_is_skipped():
    assert cited_sentences("Yes.") == []


def test_multiple_markers_in_one_sentence_are_all_kept():
    sentences = cited_sentences("Fact one [1] and fact two [3].")
    assert sentences == [("Fact one  and fact two .", [1, 3])]


def test_lexical_support_counts_content_words_only():
    # "the" and "keeps" are dropped as stopwords/short, so 3 of 5 words count.
    assert lexical_support("The rewritten chunk keeps tables intact.", "rewritten chunk tables") == 0.6
    assert lexical_support("the chunk is intact", "chunk intact") == 1.0
    assert lexical_support("The rewritten chunk keeps tables intact.", "unrelated words here") == 0.0


def test_correct_citation_is_supported():
    blocks = ["the gold passage about normalization", "a distractor about fonts", "another distractor"]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    verdict = check_sentence(
        "Cosine space needs normalized vectors.",
        [1],
        blocks,
        vectors,
        sentence_vector=[1.0, 0.0],
    )
    assert verdict.verdict == "supported"
    assert verdict.best_block == 1


def test_drifted_citation_is_misattributed():
    # Cites block 2 while the evidence sits in block 1 — the real failure mode.
    blocks = ["the gold passage about normalization", "a distractor about fonts", "another distractor"]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    verdict = check_sentence(
        "Cosine space needs normalized vectors.",
        [2],
        blocks,
        vectors,
        sentence_vector=[1.0, 0.0],
    )
    assert verdict.verdict == "misattributed"
    assert verdict.best_block == 1


def test_marker_outside_the_prompt_is_uncited():
    verdict = check_sentence("Invented source.", [9], ["only one block"], [[1.0, 0.0]], sentence_vector=[1.0, 0.0])
    assert verdict.verdict == "uncited"


def test_support_rate_ignores_uncited_sentences():
    blocks = ["gold text", "other text"]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    good = check_sentence("Gold text here.", [1], blocks, vectors, sentence_vector=[1.0, 0.0])
    gap = check_sentence("Unsupported marker.", [9], blocks, vectors, sentence_vector=[1.0, 0.0])
    assert support_rate([good, gap]) == 1.0
