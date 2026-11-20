"""ChromaDB persistence layer.

One collection holds every chunk of every document. Chunk ids are
``"<source>:<page>:<index>"``, which makes re-indexing idempotent: the previous
version of a document is deleted before its chunks are written again, so an
edited PDF cannot leave stale chunks behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import chromadb

from . import config
from .chunking import Chunk


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is better
    embedding: list[float] = field(default_factory=list, repr=False)


class DocumentStore:
    def __init__(self, path: Optional[Path | str] = None, collection_name: Optional[str] = None):
        self._client = chromadb.PersistentClient(path=str(path or config.CHROMA_PATH))
        self._name = collection_name or config.COLLECTION

    # -- collection ----------------------------------------------------------
    @property
    def collection(self):
        return self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    # -- writes -------------------------------------------------------------
    def add_document(
        self,
        source: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        ids = [f"{source}:{c.page}:{c.index}" for c in chunks]
        metadatas = [
            {"source": source, "page": c.page, "chunk": c.index, "chars": len(c.text)}
            for c in chunks
        ]
        self.collection.upsert(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=[list(e) for e in embeddings],
            metadatas=metadatas,
        )
        return len(ids)

    def delete_document(self, source: str) -> int:
        """Remove every chunk of one document and return how many were deleted."""
        existing = self.collection.get(where={"source": source}, include=[])
        deleted = len(existing.get("ids") or [])
        if deleted:
            self.collection.delete(where={"source": source})
        return deleted

    def clear(self) -> None:
        self._client.delete_collection(self._name)

    # -- reads --------------------------------------------------------------
    def documents(self) -> dict[str, dict[str, int]]:
        """Per-document chunk and page counts, for the sidebar."""
        data = self.collection.get(include=["metadatas"])
        stats: dict[str, dict] = {}
        for metadata in data.get("metadatas") or []:
            if not metadata:
                continue
            source = str(metadata.get("source", "unknown"))
            entry = stats.setdefault(source, {"chunks": 0, "pages": set()})
            entry["chunks"] += 1
            entry["pages"].add(metadata.get("page"))
        return {src: {"chunks": v["chunks"], "pages": len(v["pages"])} for src, v in stats.items()}

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        sources: Optional[Sequence[str]] = None,
    ) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []

        where = {"source": {"$in": list(sources)}} if sources else None
        result = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=max(1, min(k, total)),
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        embeddings = (result.get("embeddings") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for i, text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            distance = float(distances[i]) if i < len(distances) else 1.0
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(metadata.get("source", "unknown")),
                    page=int(metadata.get("page", 0) or 0),
                    score=1.0 - distance,  # cosine space: distance = 1 - similarity
                    embedding=list(embeddings[i]) if i < len(embeddings) and embeddings[i] is not None else [],
                )
            )
        return chunks
