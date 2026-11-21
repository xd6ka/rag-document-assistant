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
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "8"))   # candidates pulled from the vector index
CONTEXT_K = int(os.getenv("CONTEXT_K", "4"))     # chunks actually sent to the LLM
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))  # 1.0 = pure relevance, 0.0 = pure diversity

# --- reranking --------------------------------------------------------------
# Cross-encoder over the candidate set: joint (query, passage) scoring, one forward
# pass per candidate. English-only model — swap it for a multilingual checkpoint
# if the corpus stops being English.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
