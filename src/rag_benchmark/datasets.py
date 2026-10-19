"""Vector datasets and exact (brute-force) ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Metric = Literal["cosine", "l2"]
METRICS: tuple[Metric, ...] = ("cosine", "l2")

# Upper bound on the float32 score block held in memory by exact_knn.
_SCORE_BLOCK_BYTES = 256 * 1024 * 1024
# Queries scored per block (also bounds the float64 re-ranking buffer).
_MAX_QUERY_BLOCK = 256
# Extra candidates kept from the float32 pass and re-ranked in float64.
_RERANK_MARGIN = 16


@dataclass
class VectorDataset:
    """Base vectors to index plus the query vectors to search with."""

    name: str
    base: np.ndarray
    queries: np.ndarray
    metric: Metric = "cosine"

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {self.metric!r}")
        self.base = np.ascontiguousarray(self.base, dtype=np.float32)
        self.queries = np.ascontiguousarray(self.queries, dtype=np.float32)
        if self.base.ndim != 2 or self.queries.ndim != 2:
            raise ValueError("base and queries must be 2-D arrays")
        if self.base.shape[1] != self.queries.shape[1]:
            raise ValueError(
                f"dimension mismatch: base {self.base.shape[1]}, queries {self.queries.shape[1]}"
            )
        if len(self.base) == 0 or len(self.queries) == 0:
            raise ValueError("base and queries must be non-empty")

    @property
    def n(self) -> int:
        return int(self.base.shape[0])

    @property
    def dim(self) -> int:
        return int(self.base.shape[1])

    @property
    def n_queries(self) -> int:
        return int(self.queries.shape[0])


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (float32 copy). All-zero rows are left as zeros."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32, copy=False)


def make_synthetic(
    n: int = 20_000,
    dim: int = 384,
    n_queries: int = 1_000,
    n_clusters: int = 64,
    cluster_std: float = 0.05,
    seed: int = 42,
    metric: Metric = "cosine",
) -> VectorDataset:
    """Gaussian clusters around random unit-norm centres.

    Queries are drawn from the same mixture but are not copies of base vectors.
    With ``metric="cosine"`` every vector is L2-normalised, which is how most
    text-embedding models are used. The output is fully determined by ``seed``.
    """
    if n <= 0 or dim <= 0 or n_queries <= 0 or n_clusters <= 0:
        raise ValueError("n, dim, n_queries and n_clusters must be positive")
    if cluster_std < 0:
        raise ValueError("cluster_std must be >= 0")

    rng = np.random.default_rng(seed)
    centres = l2_normalize(rng.standard_normal((n_clusters, dim), dtype=np.float32))

    def sample(count: int) -> np.ndarray:
        labels = rng.integers(0, n_clusters, size=count)
        noise = rng.standard_normal((count, dim), dtype=np.float32) * np.float32(cluster_std)
        return centres[labels] + noise

    base = sample(n)
    queries = sample(n_queries)
    if metric == "cosine":
        base, queries = l2_normalize(base), l2_normalize(queries)
    name = f"synthetic n={n} dim={dim} clusters={n_clusters} std={cluster_std:g} seed={seed}"
    return VectorDataset(name=name, base=base, queries=queries, metric=metric)


def load_npy(
    path: str | Path,
    n_queries: int = 1_000,
    seed: int = 42,
    metric: Metric = "cosine",
    queries_path: str | Path | None = None,
    limit: int | None = None,
) -> VectorDataset:
    """Load base vectors from a ``.npy`` file of shape ``(n, dim)``.

    If ``queries_path`` is omitted, ``n_queries`` rows are held out of the base set
    (chosen with ``seed``) and used as queries, so queries never match themselves.
    """
    base = np.load(Path(path), allow_pickle=False)
    if base.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D array, got shape {base.shape}")
    if limit is not None:
        base = base[:limit]

    if queries_path is not None:
        queries = np.load(Path(queries_path), allow_pickle=False)
        if queries.ndim != 2:
            raise ValueError(f"{queries_path}: expected a 2-D array, got shape {queries.shape}")
        queries = queries[:n_queries]
    else:
        if n_queries >= len(base):
            raise ValueError(f"cannot hold out {n_queries} queries from {len(base)} vectors")
        order = np.random.default_rng(seed).permutation(len(base))
        queries, base = base[order[:n_queries]], base[order[n_queries:]]

    if metric == "cosine":
        base, queries = l2_normalize(base), l2_normalize(queries)
    name = f"{Path(path).name} n={len(base)} dim={base.shape[1]}"
    return VectorDataset(name=name, base=base, queries=queries, metric=metric)


def exact_knn(base: np.ndarray, queries: np.ndarray, k: int, metric: Metric) -> np.ndarray:
    """Exact top-``k`` neighbour ids for every query, shape ``(n_queries, k)``.

    Scores are computed blockwise in float32 (fast BLAS), then the best
    ``k + margin`` candidates are re-scored in float64 so near-ties at the k-th
    position are resolved precisely. Ties are broken by the lower id.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    n = base.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if metric == "cosine":
        base, queries = l2_normalize(base), l2_normalize(queries)

    n_candidates = min(n, k + _RERANK_MARGIN)
    base_sq = np.einsum("ij,ij->i", base, base) if metric == "l2" else None
    block = max(1, min(_MAX_QUERY_BLOCK, _SCORE_BLOCK_BYTES // (4 * n)))
    out = np.empty((queries.shape[0], k), dtype=np.int64)

    for start in range(0, queries.shape[0], block):
        q = queries[start : start + block]
        # Higher score = closer. For L2, ||q||^2 is constant per query and dropped.
        scores = q @ base.T
        if base_sq is not None:
            scores = 2.0 * scores - base_sq
        if n_candidates < n:
            cand = np.argpartition(-scores, n_candidates - 1, axis=1)[:, :n_candidates]
        else:
            cand = np.broadcast_to(np.arange(n), scores.shape).copy()

        exact = _similarity(base, q, cand, metric)
        order = np.lexsort((cand, -exact), axis=1)[:, :k]
        out[start : start + len(q)] = np.take_along_axis(cand, order, axis=1)
    return out


def _similarity(
    base: np.ndarray, queries: np.ndarray, ids: np.ndarray, metric: Metric
) -> np.ndarray:
    """float64 similarity (higher = closer) of each query to the rows ``ids[i]``."""
    vecs = base[ids].astype(np.float64)  # (nq, c, dim)
    q64 = queries.astype(np.float64)[:, None, :]
    if metric == "l2":
        return -np.sum((vecs - q64) ** 2, axis=2)
    return np.sum(vecs * q64, axis=2)


def similarity_scores(
    base: np.ndarray, queries: np.ndarray, ids: np.ndarray, metric: Metric
) -> np.ndarray:
    """Exact float64 similarity of every (query, id) pair; ``-inf`` where ``id < 0``.

    For cosine this is the cosine similarity, for L2 the negated squared distance,
    so in both cases higher means closer.
    """
    ids = np.asarray(ids, dtype=np.int64)
    base = np.asarray(base, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if metric == "cosine":
        base, queries = l2_normalize(base), l2_normalize(queries)
    scores = np.full(ids.shape, -np.inf, dtype=np.float64)
    for start in range(0, len(ids), _MAX_QUERY_BLOCK):
        block = ids[start : start + _MAX_QUERY_BLOCK]
        valid = block >= 0
        part = _similarity(
            base, queries[start : start + len(block)], np.where(valid, block, 0), metric
        )
        scores[start : start + len(block)] = np.where(valid, part, -np.inf)
    return scores
