"""Pinecone serverless via the ``pinecone`` SDK.

Enabled only when ``PINECONE_API_KEY`` is set. Uses ``PINECONE_INDEX`` (default
``rag-benchmark``), creating it in ``PINECONE_CLOUD`` / ``PINECONE_REGION`` if it
does not exist. Every run writes into its own namespace and deletes it afterwards;
an index created by the run is deleted too unless ``keep_index`` is true.

Pinecone is a managed service: latencies include the network round trip from this
machine to the index region, so compare them with local backends with care.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from rag_benchmark.adapters.base import VectorStore, batched, pad_ids
from rag_benchmark.datasets import Metric

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 200


class PineconeStore(VectorStore):
    """Parameters: ``keep_index`` (false), ``ready_timeout_s`` (300).

    Pinecone exposes no ANN tuning knobs, so there is a single config.
    """

    name = "pinecone"

    def __init__(
        self,
        dim: int,
        metric: Metric = "cosine",
        params: Mapping[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(dim, metric, params)
        self.index_name = os.environ.get("PINECONE_INDEX") or "rag-benchmark"
        self.namespace = f"rag-bench-{uuid.uuid4().hex[:12]}"
        self._sleep = sleep
        self._pc: Any = None
        self._index: Any = None
        self._created_index = False

    @classmethod
    def unavailable_reason(cls) -> str | None:
        try:
            import pinecone  # noqa: F401
        except ImportError:
            return 'pinecone is not installed (pip install "rag-benchmark[pinecone]")'
        if not os.environ.get("PINECONE_API_KEY", "").strip():
            return "PINECONE_API_KEY is not set"
        return None

    def _open_index(self) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self._pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"].strip())
        if not self._pc.has_index(self.index_name):
            logger.info("creating Pinecone index %s", self.index_name)
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dim,
                metric="cosine" if self.metric == "cosine" else "euclidean",
                spec=ServerlessSpec(
                    cloud=os.environ.get("PINECONE_CLOUD") or "aws",
                    region=os.environ.get("PINECONE_REGION") or "us-east-1",
                ),
            )
            self._created_index = True
        self._index = self._pc.Index(self.index_name)
        existing_dim = getattr(self._index.describe_index_stats(), "dimension", None)
        if existing_dim not in (None, self.dim):
            raise ValueError(
                f"Pinecone index {self.index_name!r} has dimension {existing_dim}, "
                f"dataset has {self.dim}; set PINECONE_INDEX to another index"
            )

    def _vector_count(self) -> int:
        namespaces = self._index.describe_index_stats().namespaces or {}
        summary = namespaces.get(self.namespace)
        return int(getattr(summary, "vector_count", 0) or 0) if summary is not None else 0

    def build(self, vectors: np.ndarray) -> None:
        x = self._as_matrix(vectors)
        self._open_index()
        for part in batched(len(x), _UPSERT_BATCH):
            records = [(str(i), x[i].tolist()) for i in range(part.start, part.stop)]
            self._index.upsert(vectors=records, namespace=self.namespace, show_progress=False)

        # Writes are eventually consistent: wait until every vector is visible.
        deadline = time.monotonic() + float(self.params.get("ready_timeout_s", 300))
        while self._vector_count() < len(x):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Pinecone did not index {len(x)} vectors in time")
            self._sleep(2.0)

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("build() must be called before search()")
        rows = []
        for q in self._as_matrix(queries):
            response = self._index.query(vector=q.tolist(), top_k=k, namespace=self.namespace)
            rows.append([int(match.id) for match in response.matches])
        return pad_ids(rows, k)

    def close(self) -> None:
        try:
            if self._index is not None:
                self._index.delete(delete_all=True, namespace=self.namespace)
            if self._created_index and not self.params.get("keep_index", False):
                self._pc.delete_index(self.index_name)
        finally:
            self._index = None
            self._pc = None
