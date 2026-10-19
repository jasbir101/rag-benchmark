"""Runs index configs against a dataset and collects one result row per config."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from rag_benchmark.adapters import IndexConfig, VectorStore, get_backend
from rag_benchmark.datasets import VectorDataset, exact_knn, similarity_scores
from rag_benchmark.metrics import (
    first_relevant_rank,
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
    summarize_latencies,
)
from rag_benchmark.rag import RelevanceLabels

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """One row of the report: a backend config measured on a dataset."""

    mode: str
    dataset: str
    backend: str
    config: str
    params: str
    status: str  # "ok" | "skipped" | "error"
    k: int
    n_vectors: int
    dim: int
    n_queries: int
    build_s: float | None = None
    recall_at_k: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    mean_ms: float | None = None
    qps: float | None = None
    batch_qps: float | None = None
    hit_rate_at_k: float | None = None
    mrr_at_k: float | None = None
    chunker: str = ""
    embedder: str = ""
    note: str = ""


def _timed_queries(
    store: VectorStore, queries: np.ndarray, k: int
) -> tuple[np.ndarray, list[float], float]:
    """One query per call, as a single client would issue them."""
    rows: list[np.ndarray] = []
    latencies: list[float] = []
    loop_start = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        ids = store.search(q[None, :], k)
        latencies.append(time.perf_counter() - t0)
        rows.append(ids[0])
    wall = time.perf_counter() - loop_start
    return np.vstack(rows), latencies, wall


def run_config(
    dataset: VectorDataset,
    config: IndexConfig,
    k: int,
    ground_truth: np.ndarray,
    truth_scores: np.ndarray | None = None,
    *,
    warmup: int = 20,
    relevance: RelevanceLabels | None = None,
    mode: str = "vector",
    chunker: str = "",
    embedder: str = "",
) -> BenchmarkResult:
    """Build, warm up, time single-client and batch search, and score one config."""
    result = BenchmarkResult(
        mode=mode,
        dataset=dataset.name,
        backend=config.backend,
        config=config.label,
        params=json.dumps(dict(config.params), sort_keys=True),
        status="ok",
        k=k,
        n_vectors=dataset.n,
        dim=dataset.dim,
        n_queries=dataset.n_queries,
        chunker=chunker,
        embedder=embedder,
    )
    store_cls = get_backend(config.backend)
    reason = store_cls.unavailable_reason()
    if reason:
        result.status, result.note = "skipped", reason
        return result

    store = None
    try:
        store = store_cls(dataset.dim, dataset.metric, config.params)
        t0 = time.perf_counter()
        store.build(dataset.base)
        result.build_s = time.perf_counter() - t0

        # Warm caches / lazy loading with base vectors, not the timed queries.
        for q in dataset.base[: min(warmup, dataset.n)]:
            store.search(q[None, :], k)

        retrieved, latencies, wall = _timed_queries(store, dataset.queries, k)
        stats = summarize_latencies(latencies, wall)
        result.p50_ms, result.p95_ms, result.p99_ms = stats.p50_ms, stats.p95_ms, stats.p99_ms
        result.mean_ms, result.qps = stats.mean_ms, stats.qps

        t0 = time.perf_counter()
        store.search(dataset.queries, k)
        batch_s = time.perf_counter() - t0
        result.batch_qps = dataset.n_queries / batch_s if batch_s > 0 else None

        if truth_scores is None:
            truth_scores = similarity_scores(
                dataset.base, dataset.queries, ground_truth, dataset.metric
            )
        result.recall_at_k = recall_at_k(
            retrieved,
            ground_truth,
            k,
            retrieved_scores=similarity_scores(
                dataset.base, dataset.queries, retrieved, dataset.metric
            ),
            truth_scores=truth_scores,
        )
        if relevance is not None:
            labels = relevance.item_labels
            ranks = [
                first_relevant_rank([labels[i] for i in row if i >= 0], target)
                for row, target in zip(retrieved, relevance.query_targets, strict=True)
            ]
            result.hit_rate_at_k = hit_rate_at_k(ranks, k)
            result.mrr_at_k = mrr_at_k(ranks, k)
    except Exception as exc:
        logger.debug("benchmark failed for %s %s", config.backend, config.label, exc_info=True)
        result.status, result.note = "error", f"{type(exc).__name__}: {exc}"
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                logger.warning("close() failed for %s", config.backend, exc_info=True)
    return result


def run_benchmark(
    dataset: VectorDataset,
    configs: Sequence[IndexConfig],
    k: int = 10,
    *,
    warmup: int = 20,
    relevance: RelevanceLabels | None = None,
    mode: str = "vector",
    chunker: str = "",
    embedder: str = "",
    progress: Callable[[BenchmarkResult], None] | None = None,
) -> list[BenchmarkResult]:
    """Run every config on ``dataset``; exact ground truth is computed once with numpy."""
    if k > dataset.n:
        raise ValueError(f"k={k} exceeds the number of indexed vectors ({dataset.n})")
    logger.info("computing exact top-%d ground truth for %d queries", k, dataset.n_queries)
    ground_truth = exact_knn(dataset.base, dataset.queries, k, dataset.metric)
    truth_scores = similarity_scores(dataset.base, dataset.queries, ground_truth, dataset.metric)
    results = []
    unavailable: set[str] = set()
    for config in configs:
        if config.backend in unavailable:
            continue  # already reported once as skipped
        result = run_config(
            dataset,
            config,
            k,
            ground_truth,
            truth_scores,
            warmup=warmup,
            relevance=relevance,
            mode=mode,
            chunker=chunker,
            embedder=embedder,
        )
        if result.status == "skipped":
            unavailable.add(config.backend)
            result.config = "all configs"
            result.params = "{}"
        if progress is not None:
            progress(result)
        results.append(result)
    return results
