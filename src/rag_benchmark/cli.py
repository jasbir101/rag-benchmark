"""Command-line interface: ``rag-bench vector``, ``rag-bench rag``, ``rag-bench backends``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_benchmark import __version__
from rag_benchmark.adapters import BACKENDS
from rag_benchmark.config import Settings, build_index_configs, load_dotenv, load_settings
from rag_benchmark.datasets import VectorDataset, load_npy, make_synthetic
from rag_benchmark.embedders import EmbeddingCache, make_embedder
from rag_benchmark.rag import build_rag_dataset, load_corpus, load_questions
from rag_benchmark.reporting import (
    plot_recall_vs_latency,
    system_info,
    write_csv,
    write_markdown,
    write_run_json,
)
from rag_benchmark.runner import BenchmarkResult, run_benchmark


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="YAML settings file (CLI flags win)")
    common.add_argument(
        "--backends", help=f"comma-separated subset of: {','.join(BACKENDS)} (default: all)"
    )
    common.add_argument("--k", type=int, help="neighbours per query (default 10)")
    common.add_argument("--warmup", type=int, help="untimed warm-up queries per config")
    common.add_argument("--out", dest="output_dir", help="output directory (default ./results)")
    common.add_argument(
        "--no-chart", dest="chart", action="store_const", const=False, help="skip the PNG chart"
    )
    common.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-bench",
        description="Benchmark vector databases for RAG retrieval: recall@k against exact "
        "ground truth, query latency, throughput and build time.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    common = _common_options()

    vec = sub.add_parser(
        "vector", parents=[common], help="synthetic or .npy vectors (offline by default)"
    )
    vec.add_argument("--n", type=int, help="base vectors (default 20000)")
    vec.add_argument("--dim", type=int, help="dimensions (default 384)")
    vec.add_argument("--clusters", type=int, help="Gaussian clusters (default 64)")
    vec.add_argument("--cluster-std", type=float, help="per-dimension cluster spread (0.05)")
    vec.add_argument("--queries", type=int, help="query vectors (default 1000)")
    vec.add_argument("--seed", type=int, help="random seed (default 42)")
    vec.add_argument("--metric", choices=["cosine", "l2"], help="distance (default cosine)")
    vec.add_argument("--npy", help="load base vectors from an (n, dim) .npy file instead")
    vec.add_argument("--queries-npy", help="query vectors .npy (default: hold out from --npy)")

    rag = sub.add_parser("rag", parents=[common], help="retrieval quality on a text corpus")
    rag.add_argument("--corpus", help="folder of .txt / .md documents")
    rag.add_argument("--questions", help='JSONL of {"question": ..., "answer_doc": ...}')
    rag.add_argument(
        "--embedder", choices=["hash", "local", "gemini"], help="embedding model (default hash)"
    )
    rag.add_argument("--embed-model", help="model name for local / gemini embedders")
    rag.add_argument("--embed-dim", type=int, help="output dimensions (hash, gemini)")
    rag.add_argument(
        "--chunker",
        dest="chunkers",
        choices=["fixed", "recursive", "all"],
        help="chunking strategy (default: all)",
    )
    rag.add_argument("--chunk-size", type=int, help="max characters per chunk (default 800)")
    rag.add_argument("--chunk-overlap", type=int, help="overlap in characters (default 100)")
    rag.add_argument("--cache-dir", help="embedding cache folder (default .cache/embeddings)")
    rag.add_argument(
        "--no-cache", dest="no_cache", action="store_true", help="do not cache embeddings"
    )

    sub.add_parser("backends", help="list backends and whether they can run here")
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    skip = {"command", "config", "verbose", "no_cache"}
    values = {k: v for k, v in vars(args).items() if k not in skip}
    if values.get("chunkers") == "all":
        values["chunkers"] = ["fixed", "recursive"]
    if getattr(args, "no_cache", False):
        values["cache_dir"] = ""
    return values


def _print_result(result: BenchmarkResult) -> None:
    head = f"{result.backend:<9} {result.config:<30}"
    if result.chunker:
        head = f"{result.chunker:<10}{head}"
    if result.status != "ok":
        print(f"{head} {result.status}: {result.note}", flush=True)
        return
    line = (
        f"{head} recall@{result.k}={result.recall_at_k:.4f}  p95={result.p95_ms:.3f} ms  "
        f"qps={result.qps:,.0f}  build={result.build_s:.2f} s"
    )
    if result.hit_rate_at_k is not None:
        line += f"  hit@{result.k}={result.hit_rate_at_k:.3f}  mrr={result.mrr_at_k:.3f}"
    print(line, flush=True)


def _vector_dataset(settings: Settings) -> VectorDataset:
    if settings.npy:
        return load_npy(
            settings.npy,
            n_queries=settings.queries,
            seed=settings.seed,
            metric=settings.metric,
            queries_path=settings.queries_npy,
        )
    return make_synthetic(
        n=settings.n,
        dim=settings.dim,
        n_queries=settings.queries,
        n_clusters=settings.clusters,
        cluster_std=settings.cluster_std,
        seed=settings.seed,
        metric=settings.metric,
    )


def run_vector(settings: Settings) -> tuple[list[BenchmarkResult], dict[str, Any]]:
    dataset = _vector_dataset(settings)
    print(f"dataset: {dataset.name}, {dataset.n_queries} queries, metric={dataset.metric}")
    results = run_benchmark(
        dataset,
        build_index_configs(settings),
        settings.k,
        warmup=settings.warmup,
        progress=_print_result,
    )
    return results, {"dataset": dataset.name}


def run_rag(settings: Settings) -> tuple[list[BenchmarkResult], dict[str, Any]]:
    if not settings.corpus or not settings.questions:
        raise ValueError("rag mode needs --corpus and --questions (or set them in --config)")
    docs = load_corpus(settings.corpus)
    questions = load_questions(settings.questions, docs)
    embedder = make_embedder(settings.embedder, settings.embed_model, settings.embed_dim)
    cache = EmbeddingCache(settings.cache_dir or None)
    corpus_name = Path(settings.corpus).name
    print(
        f"corpus: {len(docs)} documents, {len(questions)} questions, embedder={embedder.signature}"
    )

    configs = build_index_configs(settings)
    results: list[BenchmarkResult] = []
    for chunker in settings.chunkers:
        dataset, labels, _ = build_rag_dataset(
            docs,
            questions,
            chunker,
            embedder,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            cache=cache,
            corpus_name=corpus_name,
        )
        print(f"chunker={chunker}: {dataset.n} chunks, dim={dataset.dim}")
        k = min(settings.k, dataset.n)
        results += run_benchmark(
            dataset,
            configs,
            k,
            warmup=settings.warmup,
            relevance=labels,
            mode="rag",
            chunker=chunker,
            embedder=embedder.signature,
            progress=_print_result,
        )
    return results, {"corpus": settings.corpus, "embedder": embedder.signature}


def list_backends() -> int:
    for name, cls in BACKENDS.items():
        reason = cls.unavailable_reason()
        print(f"{name:<9} {'available' if reason is None else 'unavailable: ' + reason}")
    return 0


def _write_outputs(
    results: list[BenchmarkResult],
    settings: Settings,
    mode: str,
    meta: dict[str, Any],
    started: datetime,
) -> Path:
    out_dir = Path(settings.output_dir) / f"{mode}-{started:%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: list[str] = []
    if settings.chart:
        groups = sorted({r.chunker for r in results})
        for group in groups:
            name = f"recall_vs_p95_{group}.png" if group else "recall_vs_p95.png"
            subset = [r for r in results if r.chunker == group]
            title = f"Recall@{settings.k} vs p95 latency ({group} chunks)" if group else None
            if plot_recall_vs_latency(subset, out_dir / name, title=title):
                charts.append(name)
    meta = {
        "mode": mode,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "metric": settings.metric,
        "settings": vars(settings),
        "system": system_info(),
        **meta,
    }
    write_csv(results, out_dir / "results.csv")
    write_markdown(results, out_dir / "results.md", meta, charts)
    write_run_json(meta, out_dir / "run.json")
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(getattr(args, "verbose", 0), 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(Path.cwd() / ".env")

    if args.command == "backends":
        return list_backends()

    started = datetime.now()
    try:
        settings = load_settings(args.config, _overrides(args))
        if args.command == "vector":
            results, meta = run_vector(settings)
        else:
            results, meta = run_rag(settings)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = _write_outputs(results, settings, args.command, meta, started)
    print(f"\nresults written to {out_dir}")
    return 1 if any(r.status == "error" for r in results) else 0
