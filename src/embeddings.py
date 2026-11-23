"""Lazy cache of sentence-transformers models, keyed by model name.

Loading a model costs a few seconds and tens of megabytes of RAM, and Streamlit
reruns the script on every interaction, so anything that reloads it per rerun
makes the UI feel broken.

The cache key is the *model name*, not a single global slot. A one-slot cache
would let a multilingual ablation silently reuse the English model already in
memory, and every "multilingual" number would then be measured against a model
other than the one the config claims — a comparison that looks reasonable in a
report and is simply wrong.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from . import config  # noqa: F401  (sets HF_ENDPOINT before the heavy import)

_models: dict[str, object] = {}


def get_model(model_name: Optional[str] = None):
    """Return a shared ``SentenceTransformer`` for ``model_name``, loading it once."""
    name = model_name or config.EMBEDDING_MODEL
    model = _models.get(name)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name)
        _models[name] = model
    return model


def embed_texts(
    texts: Iterable[str],
    batch_size: int = 32,
    model_name: Optional[str] = None,
) -> list[list[float]]:
    """Embed a batch of texts into plain Python lists (JSON/Chroma friendly)."""
    batch = [t for t in texts]
    if not batch:
        return []
    vectors = get_model(model_name).encode(
        batch,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_text(text: str, model_name: Optional[str] = None) -> list[float]:
    return embed_texts([text], model_name=model_name)[0]


def embed_query(text: str, model_name: Optional[str] = None) -> list[float]:
    return embed_text(text, model_name=model_name)


__all__: Sequence[str] = ["get_model", "embed_texts", "embed_text", "embed_query"]
