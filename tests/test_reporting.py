import csv

import pytest

from rag_benchmark.reporting import (
    markdown_table,
    not_run_summary,
    plot_recall_vs_latency,
    system_info,
    write_csv,
    write_markdown,
)
from rag_benchmark.runner import BenchmarkResult


def _result(backend="faiss", config="flat", status="ok", **kwargs):
    base = dict(
        mode="vector",
        dataset="synthetic",
        backend=backend,
        config=config,
        params="{}",
        status=status,
        k=10,
        n_vectors=1000,
        dim=16,
        n_queries=50,
    )
    if status == "ok":
        base.update(
            build_s=0.5,
            recall_at_k=0.987654,
            p50_ms=0.1,
            p95_ms=0.25,
            p99_ms=0.4,
            mean_ms=0.12,
            qps=8123.4,
            batch_qps=50000.0,
        )
    base.update(kwargs)
    return BenchmarkResult(**base)


@pytest.fixture
def results():
    return [
        _result(),
        _result(
            config="hnsw M=16 efS=32", params='{"index": "hnsw", "ef_search": 32}', p95_ms=0.05
        ),
        _result(backend="chroma", config="M=16 efS=64", params='{"ef_search": 64}', p95_ms=0.8),
        _result(
            backend="milvus", config="all configs", status="skipped", note="MILVUS_URI is not set"
        ),
    ]


def test_write_csv(results, tmp_path):
    path = write_csv(results, tmp_path / "out" / "results.csv")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert rows[0]["recall_at_k"] == "0.987654"
    assert rows[3]["status"] == "skipped"
    assert rows[3]["recall_at_k"] == ""  # None is written as an empty cell


def test_markdown_table_formats_ok_rows_only(results):
    table = markdown_table(results)
    lines = table.strip().splitlines()
    assert lines[0].startswith("| Backend | Config | Build (s) | Recall@10 |")
    assert len(lines) == 2 + 3  # header + separator + three ok rows
    assert "| faiss | flat | 0.50 | 0.9877 | 0.100 | 0.250 | 0.400 | 8,123 | 50,000 |" in table
    assert "milvus" not in table


def test_markdown_table_adds_rag_columns():
    table = markdown_table([_result(chunker="fixed", hit_rate_at_k=0.9, mrr_at_k=0.75)])
    assert table.startswith("| Chunker | Backend |")
    assert "| 0.900 | 0.750 |" in table


def test_markdown_handles_no_successful_runs():
    assert "No successful runs" in markdown_table([_result(status="error", note="boom")])


def test_not_run_summary(results):
    assert not_run_summary(results) == "- **milvus** skipped: MILVUS_URI is not set\n"


def test_write_markdown(results, tmp_path):
    meta = {"metric": "cosine", "system": {"cpu": "Test CPU", "logical_cpus": 4, "ram_gb": 8}}
    text = write_markdown(results, tmp_path / "results.md", meta, ["chart.png"]).read_text(
        encoding="utf-8"
    )
    assert "Test CPU (4 logical cores)" in text
    assert "Not run:" in text
    assert "![recall vs p95 latency](chart.png)" in text


def test_plot_writes_png(results, tmp_path):
    path = plot_recall_vs_latency(results, tmp_path / "chart.png")
    assert path is not None
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_skips_when_nothing_succeeded(tmp_path):
    assert plot_recall_vs_latency([_result(status="skipped")], tmp_path / "c.png") is None
    assert not (tmp_path / "c.png").exists()


def test_system_info_has_the_basics():
    info = system_info()
    assert {"cpu", "logical_cpus", "ram_gb", "os", "python", "packages"} <= set(info)
    assert "numpy" in info["packages"]
