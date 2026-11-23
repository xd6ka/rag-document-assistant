"""Prompt assembly and LLM calls against any OpenAI-compatible endpoint.

The client is built from ``config.LLM_*``, so the same code talks to DashScope,
DeepSeek, OpenAI or a local Ollama server. Nothing here knows the vendor name.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

from openai import OpenAI

from . import config
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


def build_prompt(question: str, contexts: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(contexts, start=1):
        blocks.append(
            f"[{number}] source: {chunk.source}, page {chunk.page}\n{chunk.text}"
        )
    context = "\n\n".join(blocks)
    return (
        f"Context:\n{context}\n\n"
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
