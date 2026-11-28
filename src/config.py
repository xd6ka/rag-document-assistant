"""Central configuration: env loading, paths, retrieval defaults and LLM provider.

Importing this module has one side effect that matters: it sets ``HF_ENDPOINT``
before any module imports ``sentence_transformers``, so model downloads go
through the mirror when the default endpoint is unreachable.

The generation backend is resolved by :func:`resolve_llm`, kept pure so it can be
tested without touching the real environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))

# --- models -----------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Any OpenAI-compatible endpoint works. ``LLM_PROVIDER`` picks a preset and the
# ``LLM_*`` variables override individual fields, so switching vendor — or giving
# up on hosted APIs entirely — is a .env edit rather than a code change.
PROVIDER_PRESETS: dict[str, dict] = {
    "dashscope": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "QWEN_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "key_env": "MOONSHOT_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "key_env": "ZHIPU_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    # Fully offline: needs `ollama serve` and a pulled model, no key at all.
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:3b",
        "key_env": None,
    },
}
DEFAULT_PROVIDER = "ollama"
LOCAL_PROVIDERS = {"ollama"}


def resolve_llm(env: Mapping[str, str] | None = None) -> dict:
    """Resolve the generation backend from the environment.

    Precedence: explicit ``LLM_*`` value > provider preset. The legacy ``QWEN_*``
    names are honoured *only* for DashScope — otherwise a leftover QWEN_BASE_URL
    would silently point the local Ollama client at a paid endpoint.
    """
    env = os.environ if env is None else env

    explicit = (env.get("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        provider = explicit
    elif env.get("QWEN_API_KEY"):
        provider = "dashscope"
    else:
        provider = DEFAULT_PROVIDER

    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS[DEFAULT_PROVIDER])

    def legacy(name: str) -> str | None:
        return env.get(f"QWEN_{name}") if provider == "dashscope" else None

    is_local = provider in LOCAL_PROVIDERS
    key_env = preset.get("key_env")
    api_key = env.get("LLM_API_KEY") or (env.get(key_env) if key_env else None) or None

    return {
        "provider": provider,
        "base_url": env.get("LLM_BASE_URL") or legacy("BASE_URL") or preset["base_url"],
        "model": env.get("LLM_MODEL") or legacy("MODEL") or preset["model"],
        "api_key": api_key or None,
        "is_local": is_local,
        # CPU inference is slow: a long timeout and a capped answer keep the UI
        # responsive instead of hanging for minutes on one question.
        "timeout": float(env.get("LLM_TIMEOUT") or ("300" if is_local else "60")),
        "max_tokens": int(env.get("LLM_MAX_TOKENS", "400")),
    }


_LLM = resolve_llm()
LLM_PROVIDER = _LLM["provider"]
LLM_BASE_URL = _LLM["base_url"]
LLM_MODEL = _LLM["model"]
LLM_API_KEY = _LLM["api_key"]
LLM_IS_LOCAL = _LLM["is_local"]
LLM_TIMEOUT = _LLM["timeout"]
LLM_MAX_TOKENS = _LLM["max_tokens"]

# --- storage ----------------------------------------------------------------
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db")))
COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

# --- chunking ---------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "80"))

# --- retrieval --------------------------------------------------------------
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "16"))   # candidates pulled from the vector index
# 16, not 8: on real PDFs the gold page sat outside the top 8 candidates for 22 %
# of questions (eval/reports/real_depth.json), and the reranker cannot recover a
# chunk it never sees. On the synthetic corpus 8 and 16 score identically.
CONTEXT_K = int(os.getenv("CONTEXT_K", "4"))     # chunks actually sent to the LLM
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))  # 1.0 = pure relevance, 0.0 = pure diversity
# Chunks fetched from each document before the candidates are merged (0 = off).
# A global top-k is answered by the document the question resembles most, so a
# question whose evidence sits in two documents gets a prompt made of one: measured
# on eight arXiv papers, four of four context blocks came from a single PDF and the
# second paper was never retrieved, at k=4 and at k=8. A quota spends the same
# candidate budget with coverage guaranteed; it costs one index query per document.
SOURCE_QUOTA = int(os.getenv("SOURCE_QUOTA", "0"))

# --- reranking --------------------------------------------------------------
# Cross-encoder over the candidate set: joint (query, passage) scoring, one forward
# pass per candidate. English-only model — swap it for a multilingual checkpoint
# if the corpus stops being English.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

# --- OCR fallback -----------------------------------------------------------
# Scanned pages carry no text layer; when a page yields almost no text it is
# rasterised and read with RapidOCR instead. 300 dpi is the default because it
# reproduces ~99 % of a page's words in the benchmark, against 84 % at 200 dpi.
OCR_ENABLED = os.getenv("OCR_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_TRIGGER_CHARS = int(os.getenv("OCR_TRIGGER_CHARS", "60"))  # below this, try OCR
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.5"))

# --- boilerplate ------------------------------------------------------------
# Running headers and footers repeat on most pages of a real document; they are
# stripped before chunking so every chunk does not start with the same sentence.
STRIP_BOILERPLATE = os.getenv("STRIP_BOILERPLATE", "1").lower() in {"1", "true", "yes", "on"}
BOILERPLATE_PAGE_SHARE = float(os.getenv("BOILERPLATE_PAGE_SHARE", "0.4"))

# --- entailment -------------------------------------------------------------
# Sentence-level NLI answers the question the lexical citation check cannot: does
# the cited block actually *say* this? Each block is split into sentences and the
# claim is entailed if any sentence entails it, which avoids handing an 800-char
# chunk to a model trained on single-sentence pairs.
#
# Off by default: a metric ships after it is calibrated against the hand audit
# (eval/reports/entailment.json), not because an NLI checkpoint exists.
#
# The default threshold sits in the middle of the measured gap rather than on its
# edge: on the 25 audited claims every drifted sentence scores at most 0.144 and the
# weakest supported one scores 0.541, so anything in 0.15-0.50 gives the same
# accuracy and only the margin differs. The midpoint keeps both tails at arm's
# length from a single hard sentence.
ENTAIL_ENABLED = os.getenv("ENTAIL_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
ENTAIL_MODEL = os.getenv("ENTAIL_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")
ENTAIL_THRESHOLD = float(os.getenv("ENTAIL_THRESHOLD", "0.35"))

# --- prompt budget ----------------------------------------------------------
# Characters per context block sent to the model; 0 means send the block whole.
# On CPU the answer takes ~24 s and almost all of it is prompt prefill, so this is
# the one number that moves latency. Trimming keeps the sentences whose content
# words the question uses rather than cutting the block at an offset — see
# src/generation.py:trim_block and the measured trade-off in the README.
CONTEXT_BUDGET = int(os.getenv("CONTEXT_BUDGET", "400"))
