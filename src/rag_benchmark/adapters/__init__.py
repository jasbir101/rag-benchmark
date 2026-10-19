"""Backend registry. Adapters import their SDKs lazily, so importing this module is cheap."""

from __future__ import annotations

from rag_benchmark.adapters.base import (
    BackendUnavailableError,
    IndexConfig,
    VectorStore,
    pad_ids,
)
from rag_benchmark.adapters.chroma_backend import ChromaStore
from rag_benchmark.adapters.faiss_backend import FaissStore
from rag_benchmark.adapters.milvus_backend import MilvusStore
from rag_benchmark.adapters.pinecone_backend import PineconeStore

BACKENDS: dict[str, type[VectorStore]] = {
    cls.name: cls for cls in (FaissStore, ChromaStore, MilvusStore, PineconeStore)
}


def get_backend(name: str) -> type[VectorStore]:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None


__all__ = [
    "BACKENDS",
    "BackendUnavailableError",
    "ChromaStore",
    "FaissStore",
    "IndexConfig",
    "MilvusStore",
    "PineconeStore",
    "VectorStore",
    "get_backend",
    "pad_ids",
]
