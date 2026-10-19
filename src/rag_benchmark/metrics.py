"""Retrieval-quality and latency metrics.

All functions are pure and work on plain numpy arrays / Python sequences, so they
can be unit-tested without any vector database.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def recall_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    *,
    retrieved_scores: np.ndarray | None = None,
    truth_scores: np.ndarray | None = None,
    tol: float = 1e-6,
) -> float:
    """Mean recall@k of approximate results against exact nearest neighbours.

    For every query this is ``|retrieved[:k] & truth[:k]| / k``; the result is the
    mean over queries. Negative ids are treated as padding (a backend returned
    fewer than ``k`` hits) and never count as a match.

    Ties: when several vectors are exactly as close as the k-th true neighbour,
    any of them is a correct answer, but plain set intersection would count it as
    a miss. Pass exact similarity scores (higher = closer) for the retrieved ids
    and for the true top-k, and a retrieved id outside the true set also counts if
    its score is within ``tol`` of the k-th true score.
    """
    got = np.asarray(retrieved)
    truth = np.asarray(ground_truth)
    tie_aware = retrieved_scores is not None and truth_scores is not None
    if got.ndim != 2 or truth.ndim != 2:
        raise ValueError("retrieved and ground_truth must be 2-D (n_queries, k) arrays")
    if got.shape[0] != truth.shape[0]:
        raise ValueError(
            f"query count mismatch: retrieved has {got.shape[0]}, ground truth {truth.shape[0]}"
        )
    if got.shape[0] == 0:
        raise ValueError("recall is undefined for zero queries")
    if k <= 0 or truth.shape[1] < k:
        raise ValueError(f"k must be in [1, {truth.shape[1]}], got {k}")

    if tie_aware:
        scores = np.asarray(retrieved_scores, dtype=np.float64)
        if scores.shape != got.shape:
            raise ValueError("retrieved_scores must have the same shape as retrieved")
        threshold = np.asarray(truth_scores, dtype=np.float64)[:, k - 1] - tol
    else:  # nothing outside the true set can match
        scores = np.full(got.shape, -np.inf)
        threshold = np.full(got.shape[0], np.inf)

    hits = 0
    for i, (row, true_row) in enumerate(zip(got[:, :k], truth[:, :k], strict=True)):
        true_set = set(true_row.tolist())
        matched: set[int] = set()
        for j, item in enumerate(row.tolist()):
            if item >= 0 and (item in true_set or scores[i, j] >= threshold[i]):
                matched.add(item)
        hits += min(len(matched), k)
    return hits / (k * got.shape[0])


def first_relevant_rank(retrieved_labels: Sequence[str], target: str) -> int | None:
    """1-based rank of the first retrieved item whose label equals ``target``."""
    for rank, label in enumerate(retrieved_labels, start=1):
        if label == target:
            return rank
    return None


def hit_rate_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of queries with a relevant item in the top ``k``."""
    if not ranks:
        raise ValueError("hit rate is undefined for zero queries")
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Mean reciprocal rank, counting only relevant items within the top ``k``."""
    if not ranks:
        raise ValueError("MRR is undefined for zero queries")
    return sum(1.0 / r for r in ranks if r is not None and r <= k) / len(ranks)


def percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    """Percentile with linear interpolation between closest ranks (numpy's default)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    return float(np.percentile(arr, q))


@dataclass(frozen=True)
class LatencyStats:
    """Summary of per-query latencies, in milliseconds, plus throughput."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    qps: float


def summarize_latencies(
    latencies_s: Sequence[float] | np.ndarray, wall_time_s: float | None = None
) -> LatencyStats:
    """Summarise per-query latencies (given in seconds).

    ``qps`` is ``n_queries / wall_time_s`` when the wall time of the whole loop is
    known (it includes loop overhead), otherwise ``n_queries / sum(latencies)``.
    """
    arr_ms = np.asarray(latencies_s, dtype=np.float64) * 1000.0
    if arr_ms.size == 0:
        raise ValueError("no latencies to summarise")
    elapsed = wall_time_s if wall_time_s is not None else float(arr_ms.sum()) / 1000.0
    qps = arr_ms.size / elapsed if elapsed > 0 else float("inf")
    return LatencyStats(
        p50_ms=percentile(arr_ms, 50),
        p95_ms=percentile(arr_ms, 95),
        p99_ms=percentile(arr_ms, 99),
        mean_ms=float(arr_ms.mean()),
        qps=qps,
    )
