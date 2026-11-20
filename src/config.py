"""Central configuration: env loading, paths and retrieval defaults.

Importing this module has one side effect that matters: it sets ``HF_ENDPOINT``
before any module imports ``sentence_transformers``, so model downloads go
through the mirror when the default endpoint is unreachable.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))

# --- models -----------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

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
