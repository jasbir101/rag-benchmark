"""Milvus / Zilliz Cloud via ``pymilvus.MilvusClient``.

Connects to ``MILVUS_URI`` (optionally authenticated with ``MILVUS_TOKEN``). Milvus
Lite (a local ``.db`` file URI) is not available on Windows, so on Windows point
the URI at a Milvus server (e.g. the official Docker image) or Zilliz Cloud.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from rag_benchmark.adapters.base import VectorStore, batched, pad_ids
from rag_benchmark.datasets import Metric

_INDEX_TYPES = {"flat": "FLAT", "hnsw": "HNSW", "ivf_flat": "IVF_FLAT", "autoindex": "AUTOINDEX"}
_INSERT_BATCH = 1_000


class MilvusStore(VectorStore):
    """Parameters:

    - ``index``: ``hnsw`` (default), ``ivf_flat``, ``flat`` or ``autoindex``
    - HNSW: ``M`` (32), ``ef_construction`` (200), ``ef_search`` (64)
    - IVF-Flat: ``nlist`` (256), ``nprobe`` (16)

    A uniquely named collection is created per run and dropped on ``close()``.
    """

    name = "milvus"

    def __init__(
        self, dim: int, metric: Metric = "cosine", params: Mapping[str, Any] | None = None
    ):
        super().__init__(dim, metric, params)
        index = str(self.params.get("index", "hnsw")).lower()
        if index not in _INDEX_TYPES:
            raise ValueError(f"milvus index must be one of {sorted(_INDEX_TYPES)}, got {index!r}")
        self.index_type = index
        self.metric_type = "COSINE" if metric == "cosine" else "L2"
        self.collection = f"rag_bench_{uuid.uuid4().hex[:12]}"
        self._client: Any = None

    @classmethod
    def unavailable_reason(cls) -> str | None:
        try:
            import pymilvus  # noqa: F401
        except ImportError:
            return 'pymilvus is not installed (pip install "rag-benchmark[milvus]")'
        uri = os.environ.get("MILVUS_URI", "").strip()
        if not uri:
            return "MILVUS_URI is not set (Milvus server or Zilliz Cloud URI)"
        if sys.platform == "win32" and not uri.startswith(("http://", "https://", "tcp://")):
            return "Milvus Lite (local file URI) does not run on Windows; use a server URI"
        return None

    def _index_params(self) -> dict[str, Any]:
        p = self.params
        if self.index_type == "hnsw":
            return {"M": int(p.get("M", 32)), "efConstruction": int(p.get("ef_construction", 200))}
        if self.index_type == "ivf_flat":
            return {"nlist": int(p.get("nlist", 256))}
        return {}

    def _search_params(self) -> dict[str, Any]:
        p = self.params
        if self.index_type == "hnsw":
            params = {"ef": int(p.get("ef_search", 64))}
        elif self.index_type == "ivf_flat":
            params = {"nprobe": int(p.get("nprobe", 16))}
        else:
            params = {}
        return {"metric_type": self.metric_type, "params": params}

    def build(self, vectors: np.ndarray) -> None:
        from pymilvus import DataType, MilvusClient

        x = self._as_matrix(vectors)
        client = MilvusClient(
            uri=os.environ["MILVUS_URI"].strip(), token=os.environ.get("MILVUS_TOKEN", "")
        )
        self._client = client

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        client.create_collection(self.collection, schema=schema)

        for part in batched(len(x), _INSERT_BATCH):
            rows = [
                {"id": i, "vector": vec.tolist()}
                for i, vec in zip(range(part.start, part.stop), x[part], strict=True)
            ]
            client.insert(self.collection, data=rows)
        client.flush(self.collection)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=_INDEX_TYPES[self.index_type],
            metric_type=self.metric_type,
            params=self._index_params(),
        )
        client.create_index(self.collection, index_params)
        client.load_collection(self.collection)

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        if self._client is None:
            raise RuntimeError("build() must be called before search()")
        hits = self._client.search(
            self.collection,
            data=self._as_matrix(queries).tolist(),
            limit=k,
            search_params=self._search_params(),
            anns_field="vector",
        )
        return pad_ids([[int(hit["id"]) for hit in row] for row in hits], k)

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.drop_collection(self.collection)
        finally:
            self._client.close()
            self._client = None
