import csv

import numpy as np
import pytest

from rag_benchmark import adapters
from rag_benchmark.adapters import IndexConfig, VectorStore
from rag_benchmark.cli import main
from rag_benchmark.runner import run_benchmark

from .conftest import SAMPLE_DIR


class _BrokenStore(VectorStore):
    name = "broken"
    closed = False

    def build(self, vectors: np.ndarray) -> None:
        raise RuntimeError("index exploded")

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        raise AssertionError("unreachable")

    def close(self) -> None:
        _BrokenStore.closed = True


def test_runner_records_errors_and_closes(monkeypatch, small_dataset):
    monkeypatch.setitem(adapters.BACKENDS, "broken", _BrokenStore)
    [result] = run_benchmark(small_dataset, [IndexConfig("broken", "x", {})], 5, warmup=0)
    assert result.status == "error"
    assert "index exploded" in result.note
    assert _BrokenStore.closed


def test_runner_measures_everything(small_dataset):
    [result] = run_benchmark(
        small_dataset, [IndexConfig("faiss", "flat", {"index": "flat"})], 10, warmup=3
    )
    assert result.status == "ok"
    assert result.recall_at_k == 1.0
    assert result.build_s >= 0
    assert 0 < result.p50_ms <= result.p95_ms <= result.p99_ms
    assert result.qps > 0 and result.batch_qps > 0
    assert result.hit_rate_at_k is None  # vector mode has no relevance labels


def test_runner_rejects_k_larger_than_dataset(small_dataset):
    with pytest.raises(ValueError):
        run_benchmark(small_dataset, [], small_dataset.n + 1)


def test_cli_vector_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    args = ["vector", "--n", "600", "--dim", "16", "--queries", "20", "--k", "5"]
    code = main([*args, "--backends", "faiss,milvus", "--out", "out"])
    assert code == 0
    [run_dir] = list((tmp_path / "out").iterdir())
    assert {p.name for p in run_dir.iterdir()} == {
        "results.csv",
        "results.md",
        "run.json",
        "recall_vs_p95.png",
    }
    with (run_dir / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["status"] for r in rows].count("ok") == 7  # default FAISS sweep
    assert rows[-1]["backend"] == "milvus" and rows[-1]["status"] == "skipped"
    assert "skipped: MILVUS_URI is not set" in capsys.readouterr().out


def test_cli_rag_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "rag",
            "--corpus",
            str(SAMPLE_DIR / "corpus"),
            "--questions",
            str(SAMPLE_DIR / "questions.jsonl"),
            "--backends",
            "faiss",
            "--k",
            "3",
            "--chunk-size",
            "400",
            "--chunk-overlap",
            "50",
            "--no-cache",
            "--no-chart",
            "--out",
            "out",
        ]
    )
    assert code == 0
    [run_dir] = list((tmp_path / "out").iterdir())
    with (run_dir / "results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["chunker"] for r in rows} == {"fixed", "recursive"}
    assert all(r["status"] == "ok" and r["hit_rate_at_k"] for r in rows)
    assert not (tmp_path / ".cache").exists()


def test_cli_rag_requires_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["rag", "--backends", "faiss"]) == 2
    assert "--corpus" in capsys.readouterr().err


def test_cli_backends_lists_everything(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    for name in ("faiss", "chroma", "milvus", "pinecone"):
        assert name in out
