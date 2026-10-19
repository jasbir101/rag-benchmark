"""Chroma, in memory (``EphemeralClient``), with its built-in HNSW index."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from rag_benchmark.adapters.base import VectorStore, batched, pad_ids
from rag_benchmark.datasets import Metric


class ChromaStore(VectorStore):
    """Parameters (Chroma's names in brackets):

    - ``M`` [max_neighbors] (default 16), ``ef_construction`` (100), ``ef_search`` (100)
    - ``num_threads``: HNSW build threads (Chroma default: all cores)

    Each store creates its own uniquely named collection and drops it on ``close()``.
    Telemetry is disabled.
    """

    name = "chroma"

    def __init__(
        self, dim: int, metric: Metric = "cosine", params: Mapping[str, Any] | None = None
    ):
        super().__init__(dim, metric, params)
        self._client: Any = None
        self._collection: Any = None
        self._collection_name = f"rag-bench-{uuid.uuid4().hex[:12]}"

    @classmethod
    def unavailable_reason(cls) -> str | None:
        try:
            import chromadb  # noqa: F401
        except ImportError:
            return "chromadb is not installed (pip install chromadb)"
        return None

    def _hnsw_config(self) -> dict[str, Any]:
        p = self.params
        config: dict[str, Any] = {
            "space": "cosine" if self.metric == "cosine" else "l2",
            "max_neighbors": int(p.get("M", 16)),
            "ef_construction": int(p.get("ef_construction", 100)),
            "ef_search": int(p.get("ef_search", 100)),
        }
        if "num_threads" in p:
            config["num_threads"] = int(p["num_threads"])
        return config

    def build(self, vectors: np.ndarray) -> None:
        import chromadb
        from chromadb.config import Settings

        x = self._as_matrix(vectors)
        self._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self._collection = self._client.create_collection(
            name=self._collection_name,
            configuration={"hnsw": self._hnsw_config()},
            embedding_function=None,
        )
        for part in batched(len(x), self._client.get_max_batch_size()):
            ids = [str(i) for i in range(part.start, part.stop)]
            self._collection.add(ids=ids, embeddings=x[part])

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        if self._collection is None:
            raise RuntimeError("build() must be called before search()")
        response = self._collection.query(
            query_embeddings=self._as_matrix(queries), n_results=k, include=[]
        )
        return pad_ids([[int(i) for i in row] for row in response["ids"]], k)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(self._collection_name)
            finally:
                self._client = None
                self._collection = None
