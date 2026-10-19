"""The adapter interface every vector-database backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, ClassVar

import numpy as np

from rag_benchmark.datasets import METRICS, Metric, l2_normalize


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot run (SDK not installed or not configured)."""


@dataclass(frozen=True)
class IndexConfig:
    """One backend + parameter set to benchmark, e.g. FAISS HNSW with M=32."""

    backend: str
    label: str
    params: Mapping[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """A vector index that stores rows ``0..n-1`` and answers top-k queries.

    Lifecycle: ``build(vectors)`` once, then any number of ``search`` calls, then
    ``close()``. Ids returned by ``search`` are the row positions passed to
    ``build``, so results can be compared directly with exact ground truth.
    """

    #: Registry name, also used in reports.
    name: ClassVar[str]

    def __init__(
        self, dim: int, metric: Metric = "cosine", params: Mapping[str, Any] | None = None
    ):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.metric: Metric = metric
        self.params: dict[str, Any] = dict(params or {})

    @classmethod
    def unavailable_reason(cls) -> str | None:
        """Why this backend cannot run here, or ``None`` if it can."""
        return None

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Insert ``vectors`` (ids = row numbers) and make the index searchable."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return int64 ids of shape ``(n_queries, k)``, padded with ``-1``."""

    def close(self) -> None:  # noqa: B027 - optional hook, most backends need no cleanup
        """Release resources (drop temporary collections, close clients)."""

    # -- helpers shared by adapters -------------------------------------------------

    def _as_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """float32, C-contiguous, 2-D, right dimension; normalised for cosine."""
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {x.shape}")
        if self.metric == "cosine":
            x = l2_normalize(x)
        return np.ascontiguousarray(x)

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def pad_ids(rows: Sequence[Sequence[int]], k: int) -> np.ndarray:
    """Stack variable-length id lists into an ``(n, k)`` int64 array padded with -1."""
    out = np.full((len(rows), k), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        row = list(row)[:k]
        out[i, : len(row)] = row
    return out


def batched(n: int, size: int) -> Iterator[slice]:
    """Slices covering ``range(n)`` in chunks of at most ``size``."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))
