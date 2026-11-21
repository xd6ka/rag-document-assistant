"""Claim-level citation checks.

``citation_rate`` only asks whether a bracket exists. ``citation_gold_rate`` asks
whether the bracket points at the gold page, which still treats the answer as one
blob. This module goes one level down: each sentence that cites a block is checked
against the block it names, so an answer can be right as a whole and wrong per claim.

The check is evidence-based, not entailment:

- **lexical support** — how much of the sentence's content vocabulary appears in the
  cited block (a paraphrase scores low, which is expected);
- **ranking support** — the embedding of the sentence is compared with every block
  that was shown; if another block is clearly closer than the cited one, the citation
  is misattributed even when the wording overlaps.

A sentence is ``supported`` when the cited block is not clearly beaten by another
block. ``weak`` means the citation is plausible but neither signal is strong — a
reportable, not a failing, state. ``misattributed`` names the block that actually
matches best, so the report can point at the drift instead of just counting it.

No LLM judge is involved: the local 3B model already failed calibration (see
``eval/reports/judge_experiment.json``), and a metric that cannot be validated is
worse than no metric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

MARKER = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Same small stop list as the lexical metric; content words are what a citation
# should be carrying, not "the", "of", "is".
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "as", "at", "by", "from", "into", "than", "then", "so", "if", "not",
    "can", "cannot", "does", "do", "did", "has", "have", "had", "but", "which",
}


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-zа-я0-9][a-zа-я0-9-]*", text.lower())
    return [word for word in words if len(word) > 2 and word not in STOPWORDS]


def cited_sentences(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into sentences and attach the block numbers each one cites.

    Models routinely put the bracket after the final period ("... another page. [2]"),
    so a bare-marker fragment is folded into the sentence before it — otherwise the
    real claim looks uncited and only the marker gets graded. Fragments with no
    content words and no markers are dropped rather than scored as unsupported claims.
    """
    out: list[tuple[str, list[int]]] = []
    for raw in SENTENCE_SPLIT.split(answer.strip()):
        sentence = raw.strip()
        if not sentence:
            continue
        numbers = [int(match) for match in MARKER.findall(sentence)]
        text = MARKER.sub("", sentence).strip(" ,;:")
        if len(content_words(text)) < 3:
            if numbers and out:
                out[-1] = (out[-1][0], out[-1][1] + numbers)
            continue
        out.append((text, numbers))
    return out


def lexical_support(sentence: str, block: str) -> float:
    """Share of the sentence's content words that the block also contains."""
    words = content_words(sentence)
    if not words:
        return 0.0
    block_words = set(content_words(block))
    return sum(1 for word in words if word in block_words) / len(words)


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return 0.0 if denom == 0.0 else float(np.dot(va, vb) / denom)


@dataclass
class ClaimVerdict:
    sentence: str
    cited_blocks: list[int]
    best_block: int | None
    cited_overlap: float
    best_overlap: float
    cited_similarity: float
    best_similarity: float
    verdict: str

    @property
    def as_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "cited_blocks": self.cited_blocks,
            "best_block": self.best_block,
            "cited_overlap": round(self.cited_overlap, 3),
            "best_overlap": round(self.best_overlap, 3),
            "cited_similarity": round(self.cited_similarity, 3),
            "best_similarity": round(self.best_similarity, 3),
            "verdict": self.verdict,
        }


def check_sentence(
    sentence: str,
    cited_blocks: list[int],
    block_texts: Sequence[str],
    block_embeddings: Sequence[Sequence[float]],
    *,
    similarity_margin: float = 0.06,
    overlap_floor: float = 0.35,
    sentence_vector: Sequence[float] | None = None,
) -> ClaimVerdict:
    """Decide whether the citation of one sentence is attributable.

    ``similarity_margin``: a different block must beat the cited one by more than this
    before the citation is called misattributed. ``overlap_floor``: below this lexical
    overlap the sentence needs the ranking to be clearly right, not merely tied.
    ``sentence_vector`` lets a caller (or a test) supply the embedding instead of
    calling the embedder.
    """
    if sentence_vector is None and block_embeddings:
        sentence_vector = _embed(sentence)

    valid_citations = [n for n in cited_blocks if 1 <= n <= len(block_texts)]
    if not valid_citations:
        return ClaimVerdict(sentence, cited_blocks, None, 0.0, 0.0, 0.0, 0.0, "uncited")

    cited_text = " ".join(block_texts[n - 1] for n in valid_citations)
    cited_overlap = lexical_support(sentence, cited_text)
    cited_similarity = (
        max(_cos(sentence_vector, block_embeddings[n - 1]) for n in valid_citations)
        if sentence_vector is not None and block_embeddings
        else 0.0
    )

    best_index = max(
        range(len(block_texts)),
        key=lambda index: (
            lexical_support(sentence, block_texts[index]),
            _cos(sentence_vector, block_embeddings[index]) if sentence_vector is not None else 0.0,
        ),
    )
    best_block = best_index + 1
    best_overlap = lexical_support(sentence, block_texts[best_index])
    best_similarity = (
        _cos(sentence_vector, block_embeddings[best_index]) if sentence_vector is not None else 0.0
    )

    cited_is_best = best_block in valid_citations
    if cited_is_best:
        verdict = "supported" if cited_overlap >= overlap_floor or cited_overlap > best_overlap - 0.05 else "weak"
    elif best_similarity - cited_similarity > similarity_margin and cited_overlap < overlap_floor:
        verdict = "misattributed"
    elif best_overlap - cited_overlap > overlap_floor:
        verdict = "weak"
    else:
        verdict = "supported"

    return ClaimVerdict(
        sentence,
        cited_blocks,
        best_block,
        cited_overlap,
        best_overlap,
        cited_similarity,
        best_similarity,
        verdict,
    )


def _embed(text: str) -> Sequence[float]:
    from .embeddings import embed_query

    return embed_query(text)


def check_answer(
    answer: str,
    blocks: Iterable,
    **kwargs,
) -> list[ClaimVerdict]:
    """Run the check over an answer, given the retrieved blocks in prompt order."""
    block_texts = [getattr(block, "text", str(block)) for block in blocks]
    block_embeddings = [getattr(block, "embedding", None) for block in blocks]
    verdicts: list[ClaimVerdict] = []
    for sentence, numbers in cited_sentences(answer):
        if not numbers:
            verdicts.append(ClaimVerdict(sentence, [], None, 0.0, 0.0, 0.0, 0.0, "uncited"))
            continue
        verdicts.append(
            check_sentence(sentence, numbers, block_texts, block_embeddings, **kwargs)
        )
    return verdicts


def support_rate(verdicts: Sequence[ClaimVerdict]) -> float | None:
    graded = [v for v in verdicts if v.verdict != "uncited"]
    if not graded:
        return None
    good = sum(1 for v in graded if v.verdict == "supported")
    return round(good / len(graded), 3)
