"""Lazy singleton around sentence-transformers.

The model is loaded once per process: it costs ~5 s and ~90 MB of RAM, and
Streamlit reruns the script on every interaction, so anything that reloads it
per rerun makes the UI feel broken.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from . import config  # noqa: F401  (sets HF_ENDPOINT before the heavy import)

_model = None


def get_model():
    """Return the shared ``SentenceTransformer`` instance, loading it on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts into plain Python lists (JSON/Chroma friendly)."""
    batch = [t for t in texts]
    if not batch:
        return []
    vectors = get_model().encode(
        batch,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_query(text: str) -> list[float]:
    return embed_text(text)


__all__: Sequence[str] = ["get_model", "embed_texts", "embed_text", "embed_query"]
