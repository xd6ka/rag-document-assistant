"""Prompt assembly and LLM calls against any OpenAI-compatible endpoint.

The client is built from ``config.LLM_*``, so the same code talks to DashScope,
DeepSeek, OpenAI or a local Ollama server. Nothing here knows the vendor name.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

from openai import OpenAI

from . import config
from .citations import content_words, lexical_support
from .entailment import split_premise
from .store import RetrievedChunk

SYSTEM_PROMPT = (
    "You answer questions about documents.\n"
    "Rules:\n"
    "1. Use ONLY the context blocks below. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, say you do not know based on the document.\n"
    "3. Answer in the language of the question.\n"
    "4. Every factual sentence must end with the number of the block it came from, in square "
    "brackets: [1]. For two blocks use [1][2]. An answer without a bracketed number is wrong.\n"
    "Example: all-MiniLM-L6-v2 has 22 million parameters [1]."
)

# Ollama ignores the key but the SDK refuses to start without one.
LOCAL_API_KEY = "ollama"

_client: Optional[OpenAI] = None


def provider_label() -> str:
    """Short 'provider · model' string for the UI and eval reports."""
    return f"{config.LLM_PROVIDER} · {config.LLM_MODEL}"


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = config.LLM_API_KEY
        if not api_key:
            if not config.LLM_IS_LOCAL:
                raise RuntimeError(
                    f"No API key for provider '{config.LLM_PROVIDER}'. "
                    "Set LLM_API_KEY (or the provider-specific key) in .env, "
                    "or switch to LLM_PROVIDER=ollama for offline local inference."
                )
            api_key = LOCAL_API_KEY
        _client = OpenAI(
            api_key=api_key,
            base_url=config.LLM_BASE_URL,
            timeout=config.LLM_TIMEOUT,
            # Local models often answer a single request at a time; retrying inside
            # the SDK just multiplies the wall-clock cost.
            max_retries=0 if config.LLM_IS_LOCAL else 2,
        )
    return _client


def trim_block(text: str, question: str, budget: int) -> str:
    """Keep the sentences of a block that are about the question, within ``budget``.

    The prompt is the latency lever on CPU — prefill dominates, and prefill is
    proportional to characters — but truncating a block at an arbitrary offset
    cuts the answer out of the middle of the paragraph. Sentences are ranked by how
    many of their content words the question uses, the best ones are kept until the
    budget is spent, and the result is re-ordered as it appeared so the block still
    reads like prose. Scoring is lexical on purpose: it runs on every query and a
    cross-encoder pass over four blocks would cost more than it saves.
    """
    if budget <= 0 or len(text) <= budget:
        return text
    sentences = split_premise(text)
    if len(sentences) < 2 or not content_words(question):
        return text[:budget]

    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (-lexical_support(sentences[index], question), index),
    )
    keep: list[int] = []
    used = 0
    for index in ranked:
        length = len(sentences[index]) + 1
        if used + length <= budget:
            keep.append(index)
            used += length
    if not keep:
        keep = [ranked[0]]
    selected = " ".join(sentences[index] for index in sorted(keep))
    return selected


def build_prompt(
    question: str,
    contexts: Sequence[RetrievedChunk],
    *,
    budget: Optional[int] = None,
) -> str:
    limit = config.CONTEXT_BUDGET if budget is None else budget
    blocks = []
    for number, chunk in enumerate(contexts, start=1):
        text = trim_block(chunk.text, question, limit)
        blocks.append(f"[{number}] {text}")
    context = "\n\n".join(blocks)
    # The file name and page live in a list *after* the text, not in front of it.
    # A header glued to the first sentence is read as part of that sentence by a 3B
    # model: "…is called source: prompt-design.pdf, page 3 Context dilution [1]".
    # Citations only need the bracketed number, and the UI reads the page from the
    # chunk metadata, so the mapping belongs at the end.
    sources = "\n".join(f"[{number}] {chunk.source}, page {chunk.page}" for number, chunk in enumerate(contexts, start=1))
    return (
        f"Context:\n{context}\n\n"
        f"Sources:\n{sources}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. End every sentence with the block number, e.g. [1]."
    )


def stream_answer(
    question: str,
    contexts: Sequence[RetrievedChunk],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
):
    """Yield the answer token by token as the model produces it.

    A local 3B model needs ~45 s for a full answer, and all of that used to be
    spent behind a spinner with nothing on screen. Streaming does not make the
    model faster — it makes the wait visible, which is the difference between
    "broken" and "working" from the user's side.

    No retry loop here on purpose: once tokens have been handed to the UI,
    replaying the request would append a second answer to the first.
    """
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, contexts)},
    ]

    stream = client.chat.completions.create(
        model=model or config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta


def answer_question(
    question: str,
    contexts: Sequence[RetrievedChunk],
    model: Optional[str] = None,
    retries: int = 3,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, contexts)},
    ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model or config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    return ""
