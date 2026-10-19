"""FAISS (faiss-cpu): exact Flat, HNSW and IVF-Flat indexes, in process."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from rag_benchmark.adapters.base import VectorStore
from rag_benchmark.datasets import Metric

INDEX_TYPES = ("flat", "hnsw", "ivf_flat")


def resolve_nlist(value: Any, n: int) -> int:
    """``nlist`` for IVF. ``"auto"`` = 4*sqrt(n), capped so each list gets >= 39 points.

    FAISS warns when k-means has fewer than 39 training points per centroid.
    """
    if value in (None, "auto"):
        nlist = round(4 * math.sqrt(n))
        return max(1, min(nlist, n // 39 if n >= 39 else 1))
    nlist = int(value)
    if nlist <= 0:
        raise ValueError(f"nlist must be positive, got {value!r}")
    if nlist > n:
        raise ValueError(f"nlist={nlist} is larger than the number of vectors ({n})")
    return nlist


class FaissStore(VectorStore):
    """Parameters:

    - ``index``: ``flat`` (exact), ``hnsw`` or ``ivf_flat``
    - HNSW: ``M`` (default 32), ``ef_construction`` (200), ``ef_search`` (64)
    - IVF-Flat: ``nlist`` (``"auto"``), ``nprobe`` (16)

    Cosine similarity is implemented as inner product on L2-normalised vectors.
    """

    name = "faiss"

    def __init__(
        self, dim: int, metric: Metric = "cosine", params: Mapping[str, Any] | None = None
    ):
        super().__init__(dim, metric, params)
        self.index_type = str(self.params.get("index", "flat")).lower()
        if self.index_type not in INDEX_TYPES:
            raise ValueError(f"faiss index must be one of {INDEX_TYPES}, got {self.index_type!r}")
        self._index: Any = None
        self._quantizer: Any = None  # keep a Python reference alive for IVF

    @classmethod
    def unavailable_reason(cls) -> str | None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            return "faiss-cpu is not installed (pip install faiss-cpu)"
        return None

    def build(self, vectors: np.ndarray) -> None:
        import faiss

        x = self._as_matrix(vectors)
        metric = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
        p = self.params

        if self.index_type == "flat":
            index = faiss.IndexFlat(self.dim, metric)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, int(p.get("M", 32)), metric)
            index.hnsw.efConstruction = int(p.get("ef_construction", 200))
        else:
            nlist = resolve_nlist(p.get("nlist", "auto"), len(x))
            self._quantizer = faiss.IndexFlat(self.dim, metric)
            index = faiss.IndexIVFFlat(self._quantizer, self.dim, nlist, metric)
            index.train(x)

        index.add(x)

        if self.index_type == "hnsw":
            index.hnsw.efSearch = int(p.get("ef_search", 64))
        elif self.index_type == "ivf_flat":
            index.nprobe = int(p.get("nprobe", 16))
        self._index = index

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("build() must be called before search()")
        _, ids = self._index.search(self._as_matrix(queries), k)
        return ids.astype(np.int64, copy=False)

    def close(self) -> None:
        self._index = None
        self._quantizer = None
