"""Prompt assembly and LLM calls (Qwen via the OpenAI-compatible endpoint)."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from openai import OpenAI

from . import config
from .store import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. "
    "Answer using ONLY the provided context blocks. "
    "If the context does not contain the answer, reply that you do not know based on the document "
    "instead of guessing. "
    "Answer in the language of the question. "
    "Cite the blocks you relied on as [1], [2], ... right after the supported sentence."
)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.QWEN_API_KEY:
            raise RuntimeError(
                "QWEN_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        _client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)
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
        "Answer the question using only the context above."
    )


def answer_question(
    question: str,
    contexts: Sequence[RetrievedChunk],
    model: Optional[str] = None,
    retries: int = 3,
    temperature: float = 0.2,
) -> str:
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, contexts)},
    ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model or config.QWEN_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    return ""
