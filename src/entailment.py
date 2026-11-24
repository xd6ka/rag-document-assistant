"""Entailment support: does the cited block actually *say* this?

``citations.check_answer`` asks an attribution question — is the cited block the
best match for the sentence among the blocks in the prompt. It can pass a sentence
that names the right block and then asserts something the block never says, and
that gap is the one thing the current metrics openly do not cover.

This module closes it with a small NLI cross-encoder. Two details matter:

- **The premise is split into sentences.** A 150-word chunk fed to a sentence-pair
  model drifts to ``neutral`` even when the fact is inside it, because the model is
  trained on single sentences; the claim is entailed if *any* sentence of the block
  entails it (max over sentences).
- **The model is optional and late.** Importing this module costs nothing, the
  checkpoint loads on first use, and every failure (no download, broken install)
  surfaces as :class:`EntailmentUnavailable` so a caller can degrade to the lexical
  check instead of crashing an evaluation.

The label order comes from the checkpoint (``contradiction, entailment, neutral``
for the shipped model), so :func:`entail_max` reads it from the model config rather
than assuming index 1.
"""

from __future__ import annotations

import functools
import re
from typing import Callable, Iterable, Sequence

from . import config

PREMISE_SPLIT = re.compile(r"(?<=[.!?])\s+")
WHITESPACE = re.compile(r"\s+")
LONG_PIECE = 320
MIN_PREMISE_CHARS = 15
CONTEXT_WINDOW = 2

# A sentence that talks about the answer instead of about the corpus ("the title is
# not mentioned in the provided context") cannot be entailed by any block, so scoring
# it only manufactures a false accusation. These are skipped before the check.
META_CLAIM = re.compile(
    r"\b(not mentioned|does not mention|no information|not stated|provided context|"
    r"the context (?:does|do) not|не упомянут|не указан)",
    re.IGNORECASE,
)


class EntailmentUnavailable(RuntimeError):
    """Raised when the NLI model cannot be loaded or run."""


def split_premise(text: str) -> list[str]:
    """Split a retrieved block into candidate premise sentences.

    Whitespace is normalised first, and only sentence-ending punctuation splits:
    chunk text arrives hard-wrapped at about 60 characters, so treating ``\\n`` as a
    boundary turns "reaches 59.8 and bge-small-en-v1.5 reaches 62.2." into two
    meaningless fragments and every claim against them scores ``neutral``. Colons
    and semicolons stay inside a sentence for the same reason — "average: 62.2"
    must not become "average:" plus a dropped number.

    Pieces longer than :data:`LONG_PIECE` are split again on semicolons, which is
    the config-style text where one sentence lists several settings. Fragments
    shorter than :data:`MIN_PREMISE_CHARS` are dropped; if nothing survives, the
    whole text is used as one premise rather than returning nothing.
    """
    normalised = WHITESPACE.sub(" ", (text or "")).strip()
    if not normalised:
        return []
    pieces: list[str] = []
    for sentence in PREMISE_SPLIT.split(normalised):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > LONG_PIECE and ";" in sentence:
            pieces.extend(part.strip() for part in sentence.split(";") if part.strip())
        else:
            pieces.append(sentence)
    keep = [piece for piece in pieces if len(piece) >= MIN_PREMISE_CHARS]
    return keep or [normalised]


def is_meta_claim(sentence: str) -> bool:
    """True for sentences about the answer's own coverage, not about the corpus."""
    return bool(META_CLAIM.search(sentence or ""))


def premise_windows(text: str, window: int = CONTEXT_WINDOW) -> list[str]:
    """Sentences of a block plus the two-sentence windows around them.

    A single sentence is often not self-contained: "the defaults are M=16" does not
    mention ChromaDB, and "it hides how far down the answer was" replaces hit@k with
    a pronoun. Both read as ``neutral`` on their own while the block clearly supports
    the claim. Pairing every sentence with its neighbour restores the antecedent, so
    the caller takes the max over single sentences *and* windows.
    """
    sentences = split_premise(text)
    candidates = list(sentences)
    if window > 1:
        for start in range(len(sentences) - window + 1):
            joined = " ".join(sentences[start : start + window])
            if len(joined) >= MIN_PREMISE_CHARS:
                candidates.append(joined)
    return [candidate for candidate in candidates if len(candidate) >= MIN_PREMISE_CHARS] or sentences


@functools.lru_cache(maxsize=2)
def _model(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu", max_length=512)


def _label_ids(model) -> dict[str, int]:
    """Map ``entailment``/``neutral``/``contradiction`` to output indices."""
    cfg = getattr(model, "config", None)
    if cfg is None or not getattr(cfg, "id2label", None):
        cfg = getattr(getattr(model, "model", None), "config", None)
    mapping = getattr(cfg, "id2label", None) or {0: "contradiction", 1: "entailment", 2: "neutral"}
    return {str(label).lower(): int(index) for index, label in mapping.items()}


def _entail_index(model) -> int:
    for label, index in _label_ids(model).items():
        if "entail" in label:
            return index
    return 1


def available(model_name: str | None = None) -> bool:
    """True when the checkpoint can be loaded — used to skip work, not to decide."""
    try:
        _model(model_name or config.ENTAIL_MODEL)
        return True
    except Exception:  # pragma: no cover - depends on the machine, not the code
        return False


def entail_max(
    premise: str,
    hypothesis: str,
    *,
    model_name: str | None = None,
) -> float:
    """Highest entailment probability over the premise's sentences, in ``[0, 1]``."""
    candidates = premise_windows(premise)
    if not candidates or not (hypothesis or "").strip():
        return 0.0
    model = _model(model_name or config.ENTAIL_MODEL)
    try:
        raw = model.predict([[candidate, hypothesis] for candidate in candidates], apply_softmax=True)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
        raise EntailmentUnavailable(str(exc)) from exc
    index = _entail_index(model)
    return round(float(max(float(row[index]) for row in raw)), 4)


def supports(
    premise: str,
    hypothesis: str,
    *,
    threshold: float | None = None,
    scorer: Callable[[str, str], float] | None = None,
) -> bool:
    """Whether the block entails the claim at the configured threshold.

    ``scorer`` exists so tests and calibration can inject a cheap stand-in; the
    default is :func:`entail_max` with the configured checkpoint.
    """
    limit = config.ENTAIL_THRESHOLD if threshold is None else threshold
    score = (scorer or entail_max)(premise, hypothesis)
    return score >= limit


def sentence_scores(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    model_name: str | None = None,
) -> list[float]:
    """Convenience wrapper: one score per (premise, hypothesis) pair, in order."""
    return [entail_max(premise, hypothesis, model_name=model_name) for premise, hypothesis in zip(premises, hypotheses)]
